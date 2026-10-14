import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any

from bson import Binary, ObjectId
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from pymongo.errors import DuplicateKeyError

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from auth import create_access_token, get_current_user, hash_password, verify_password
from database import (
    MONGODB_DB_NAME,
    applications_collection,
    check_connection,
    client,
    resumes_collection,
    saved_jobs_collection,
    users_collection,
    workflow_sessions_collection,
)
from graph import (
    build_graph,
    extract_resume,
    is_approved,
    ranking_agent,
    resume_agent,
    search_jobs,
)

app = FastAPI(title='AI Job Search Assistant API')

frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:5173')
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url, 'http://127.0.0.1:5173'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)



def make_checkpointer():
    """Persist paused workflows in MongoDB so they survive restarts.

    Needs: pip install langgraph-checkpoint-mongodb
    """
    try:
        from langgraph.checkpoint.mongodb import MongoDBSaver
        return MongoDBSaver(client, db_name=MONGODB_DB_NAME)
    except ImportError:
        logging.warning(
            'langgraph-checkpoint-mongodb not installed; using in-memory '
            'checkpointer .'
        )
        return MemorySaver()


graph = build_graph(make_checkpointer())


class SignupRequest(BaseModel):
    name: str = Field(min_length=1)
    email: str
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: str
    password: str


class Location(BaseModel):
    city: str = ''
    state: str = ''
    country: str = ''


class SocialLinks(BaseModel):
    github: str = ''
    linkedin: str = ''


class ProfileUpdate(BaseModel):
    name: str | None = None
    phone: str | None = None
    location: Location | None = None
    headline: str | None = None
    bio: str | None = None
    social_links: SocialLinks | None = None


class WorkflowStartRequest(BaseModel):
    resume_id: str


class WorkflowMessage(BaseModel):
    thread_id: str
    # str for approve/feedback/job number; dict for preferences,
    # e.g. {"role": "Backend Engineer", "location": "Bangalore"}
    message: str | dict[str, Any]


class JobSearchRequest(BaseModel):
    query: str
    location: str


class SavedJobRequest(BaseModel):
    job_id: str
    job: dict[str, Any]


class ApplicationCreate(BaseModel):
    job: dict[str, Any]
    tailored_resume: str = ''
    cover_letter: str = ''
    status: str = 'Resume Ready'


class ApplicationStatusUpdate(BaseModel):
    status: str


def now():
    return datetime.now(timezone.utc)


def public_user(user):
    return {
        'id': str(user['_id']),
        'name': user.get('name', ''),
        'email': user.get('email', ''),
        'phone': user.get('phone', ''),
        'location': user.get('location', {}),
        'headline': user.get('headline', ''),
        'bio': user.get('bio', ''),
        'social_links': user.get('social_links', {}),
        'skills': user.get('skills', []),
        'experience': user.get('experience', []),
        'education': user.get('education', []),
        'projects': user.get('projects', []),
        'certifications': user.get('certifications', []),
        'suitable_roles': user.get('suitable_roles', []),
        'created_at': user.get('created_at'),
        'updated_at': user.get('updated_at'),
    }


def profile_for_llm(user):
    """Profile dict that is safe to put in graph state / prompts."""
    profile = public_user(user)
    profile.pop('created_at', None)
    profile.pop('updated_at', None)
    return profile


def workflow_config(thread_id: str):
    return {'configurable': {'thread_id': thread_id}}


# graph interrupt type -> input_type the frontend switches on
INPUT_TYPES = {
    'preferences': 'preferences',
    'job_selection': 'job_selection',
    'resume_review': 'resume_feedback',
    'final_review': 'final_approval',
}


def build_workflow_response(thread_id: str, result=None):
    state = graph.get_state(workflow_config(thread_id))
    values = state.values or {}

    # Pending interrupts live on the checkpoint, so GET /workflow/{id} is
    # correct too (not just the response of the call that triggered them).
    interrupts = [i for task in state.tasks for i in task.interrupts]
    if not interrupts and result:
        interrupts = list(result.get('__interrupt__', []))

    payload = None
    input_type = None
    question = None

    if interrupts:
        payload = interrupts[0].value
        if isinstance(payload, dict):
            kind = payload.get('type')
            input_type = INPUT_TYPES.get(kind, kind)
            question = payload.get('message')
        else:
            question = str(payload)

    return {
        'thread_id': thread_id,
        'status': 'waiting_for_input' if interrupts else 'completed',
        'question': question,
        'input_type': input_type,
        'payload': payload,
        'jobs': values.get('ranked_jobs', []),
        'selected_job': values.get('selected_job', {}),
        'tailored_resume': values.get('tailored_resume', ''),
        'cover_letter': values.get('cover_letter', ''),
    }


def require_session(thread_id: str, user_id: str):
    session = workflow_sessions_collection.find_one({
        'thread_id': thread_id,
        'user_id': user_id,
    })
    if not session:
        raise HTTPException(status_code=404, detail='Workflow session not found')
    return session


def save_application_from_state(user_id: str, thread_id: str):
    state = graph.get_state(workflow_config(thread_id)).values
    if not is_approved(state.get('approval', '')):
        return None

    job = state.get('selected_job', {})
    if not job:
        return None

    existing = applications_collection.find_one({
        'user_id': user_id,
        'workflow_thread_id': thread_id,
    })
    if existing:
        return existing

    document = {
        'user_id': user_id,
        'workflow_thread_id': thread_id,
        'job': job,
        'tailored_resume': state.get('tailored_resume', ''),
        'cover_letter': state.get('cover_letter', ''),
        'status': 'Resume Ready',
        'created_at': now(),
        'updated_at': now(),
    }
    result = applications_collection.insert_one(document)
    document['_id'] = result.inserted_id
    return document


@app.get('/')
def home():
    return {'message': 'AI Job Search Assistant backend is running'}


@app.get('/health')
def health():
    try:
        check_connection()
        return {'status': 'ok', 'database': 'connected'}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f'Database connection failed: {e}')


@app.post('/auth/signup')
def signup(request: SignupRequest):
    email = request.email.strip().lower()
    if users_collection.find_one({'email': email}):
        raise HTTPException(status_code=409, detail='Email is already registered')

    user = {
        '_id': str(ObjectId()),
        'name': request.name.strip(),
        'email': email,
        'password_hash': hash_password(request.password),
        'phone': '',
        'location': {},
        'preferred_roles': [],
        'preferred_locations': [],
        'work_mode': '',
        'experience_level': '',
        'salary_expectations': '',
        'created_at': now(),
        'updated_at': now(),
    }

    try:
        users_collection.insert_one(user)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail='Email is already registered')

    return {
        'access_token': create_access_token(user['_id']),
        'token_type': 'bearer',
        'user': public_user(user),
    }


@app.post('/auth/login')
def login(request: LoginRequest):
    email = request.email.strip().lower()
    user = users_collection.find_one({'email': email})

    if not user or not verify_password(request.password, user['password_hash']):
        raise HTTPException(status_code=401, detail='Invalid email or password')

    return {
        'access_token': create_access_token(str(user['_id'])),
        'token_type': 'bearer',
        'user': public_user(user),
    }


@app.get('/auth/me')
def auth_me(user=Depends(get_current_user)):
    return public_user(user)


@app.get('/profile')
def get_profile(user=Depends(get_current_user)):
    return public_user(user)


@app.put('/profile')
def update_profile(request: ProfileUpdate, user=Depends(get_current_user)):
    update_data = request.model_dump(exclude_none=True)
    if update_data:
        update_data['updated_at'] = now()
        users_collection.update_one({'_id': user['_id']}, {'$set': update_data})

    return public_user(users_collection.find_one({'_id': user['_id']}))


@app.post('/resume/upload')
async def upload_resume(file: UploadFile = File(...), user=Depends(get_current_user)):
    if not file.filename:
        raise HTTPException(status_code=400, detail='Resume file is required')
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail='Only PDF resumes are supported')

    pdf_bytes = await file.read()
    if len(pdf_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail='Resume must be 10 MB or smaller')

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name

        resume_text = extract_resume(temp_path)
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail='Could not extract text from resume')

        profile_result = resume_agent({
            'resume': resume_text,
            'profile': {}
        })
        profile = profile_result.get('profile', {})

        users_collection.update_one(
            {'_id': user['_id']},
            {
                '$set': {
                    'name': profile.get('name', user.get('name', '')),
                    'phone': profile.get('phone', user.get('phone', '')),
                    'location': profile.get('location', user.get('location', {})),
                    'headline': profile.get('headline', ''),
                    'bio': profile.get('bio', ''),
                    'social_links': profile.get(
                        'social_links',
                        user.get('social_links', {})
                    ),
                    'skills': profile.get('skills', []),
                    'experience': profile.get('experience', []),
                    'education': profile.get('education', []),
                    'projects': profile.get('projects', []),
                    'certifications': profile.get('certifications', []),
                    'suitable_roles': profile.get('suitable_roles', []),
                    'updated_at': now(),
                }
            }
        )

        updated_user = users_collection.find_one({'_id': user['_id']})

        existing = resumes_collection.find_one({'user_id': user['_id']})
        document = {
            'user_id': user['_id'],
            'filename': file.filename,
            'content_type': 'application/pdf',
            'pdf': Binary(pdf_bytes),
            'resume_text': resume_text,
            'uploaded_at': now(),
            'updated_at': now(),
        }

        if existing:
            resumes_collection.replace_one({'_id': existing['_id']}, document)
            resume_id = existing['_id']
        else:
            result = resumes_collection.insert_one(document)
            resume_id = result.inserted_id

        return {
            'resume_id': str(resume_id),
            'filename': file.filename,
            'profile': public_user(updated_user),
            'status': 'ready',
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get('/resume')
def get_resume(user=Depends(get_current_user)):
    resume = resumes_collection.find_one({'user_id': user['_id']})
    if not resume:
        return {'exists': False, 'resume': None}

    return {
        'exists': True,
        'resume': {
            'id': str(resume['_id']),
            'filename': resume.get('filename', ''),
            'uploaded_at': resume.get('uploaded_at'),
            'updated_at': resume.get('updated_at'),
        },
    }


@app.get('/resume/file')
def get_resume_file(user=Depends(get_current_user)):
    resume = resumes_collection.find_one({'user_id': user['_id']})
    if not resume:
        raise HTTPException(status_code=404, detail='Resume not found')

    return Response(
        content=bytes(resume['pdf']),
        media_type='application/pdf',
        headers={
            'Content-Disposition': f'inline; filename="{resume["filename"]}"'
        },
    )


@app.delete('/resume')
def delete_resume(user=Depends(get_current_user)):
    resumes_collection.delete_one({'user_id': user['_id']})
    return {'message': 'Resume deleted'}


@app.post('/jobs/search')
def jobs_search(request: JobSearchRequest, user=Depends(get_current_user)):
    profile = profile_for_llm(user)

    preferences = {
        'role': request.query,
        'location': request.location,
    }

    jobs = search_jobs(request.query, request.location)
    ranked = ranking_agent({
        'jobs': jobs,
        'profile': profile,
        'preferences': str(preferences),
    })

    return {'jobs': ranked.get('ranked_jobs', [])}


@app.get('/saved-jobs')
def get_saved_jobs(user=Depends(get_current_user)):
    saved = list(saved_jobs_collection.find({'user_id': user['_id']}).sort('saved_at', -1))
    return {
        'jobs': [
            {
                'id': str(item['_id']),
                'job_id': item.get('job_id'),
                'job': item.get('job', {}),
                'saved_at': item.get('saved_at'),
            }
            for item in saved
        ]
    }


@app.post('/saved-jobs')
def save_job(request: SavedJobRequest, user=Depends(get_current_user)):
    try:
        result = saved_jobs_collection.insert_one({
            'user_id': user['_id'],
            'job_id': request.job_id,
            'job': request.job,
            'saved_at': now(),
        })
    except DuplicateKeyError:
        return {'message': 'Job already saved'}

    return {'message': 'Job saved', 'id': str(result.inserted_id)}


@app.delete('/saved-jobs/{job_id}')
def delete_saved_job(job_id: str, user=Depends(get_current_user)):
    result = saved_jobs_collection.delete_one({'user_id': user['_id'], 'job_id': job_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail='Saved job not found')
    return {'message': 'Job removed'}


@app.post('/workflow/start')
def start_workflow(request: WorkflowStartRequest, user=Depends(get_current_user)):
    try:
        resume_object_id = ObjectId(request.resume_id)
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid resume_id')

    resume = resumes_collection.find_one({
        '_id': resume_object_id,
        'user_id': user['_id'],
    })
    if not resume:
        raise HTTPException(status_code=404, detail='Resume not found')

    thread_id = str(ObjectId())
    initial_state = {
        'resume': resume['resume_text'],
        'profile': profile_for_llm(user),
        'preferences': '',
        'search_query': '',
        'search_location': '',
        'jobs': [],
        'ranked_jobs': [],
        'selected_job': {},
        'tailored_resume': '',
        'feedback': '',
        'cover_letter': '',
        'approval': '',
        'application_status': '',
    }

    result = graph.invoke(initial_state, config=workflow_config(thread_id))

    workflow_sessions_collection.insert_one({
        'thread_id': thread_id,
        'user_id': user['_id'],
        'resume_id': resume['_id'],
        'created_at': now(),
        'updated_at': now(),
    })

    return build_workflow_response(thread_id, result)


@app.post('/workflow/message')
def workflow_message(request: WorkflowMessage, user=Depends(get_current_user)):
    require_session(request.thread_id, user['_id'])

    result = graph.invoke(
        Command(resume=request.message),
        config=workflow_config(request.thread_id),
    )

    workflow_sessions_collection.update_one(
        {'thread_id': request.thread_id, 'user_id': user['_id']},
        {'$set': {'updated_at': now()}},
    )

    response = build_workflow_response(request.thread_id, result)

    if response['status'] == 'completed':
        application = save_application_from_state(user['_id'], request.thread_id)
        if application:
            response['application_id'] = str(application['_id'])

    return response


@app.get('/workflow/{thread_id}')
def get_workflow(thread_id: str, user=Depends(get_current_user)):
    require_session(thread_id, user['_id'])
    return build_workflow_response(thread_id)


@app.post('/applications')
def create_application(request: ApplicationCreate, user=Depends(get_current_user)):
    document = {
        'user_id': user['_id'],
        'job': request.job,
        'tailored_resume': request.tailored_resume,
        'cover_letter': request.cover_letter,
        'status': request.status,
        'created_at': now(),
        'updated_at': now(),
    }
    result = applications_collection.insert_one(document)
    return {'id': str(result.inserted_id), 'message': 'Application created'}


@app.get('/applications')
def get_applications(user=Depends(get_current_user)):
    applications = list(applications_collection.find({'user_id': user['_id']}).sort('created_at', -1))
    return {
        'applications': [
            {
                'id': str(item['_id']),
                'job': item.get('job', {}),
                'tailored_resume': item.get('tailored_resume', ''),
                'cover_letter': item.get('cover_letter', ''),
                'status': item.get('status', 'Resume Ready'),
                'created_at': item.get('created_at'),
                'updated_at': item.get('updated_at'),
            }
            for item in applications
        ]
    }


@app.get('/applications/{application_id}')
def get_application(application_id: str, user=Depends(get_current_user)):
    try:
        object_id = ObjectId(application_id)
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid application id')

    application = applications_collection.find_one({
        '_id': object_id,
        'user_id': user['_id'],
    })
    if not application:
        raise HTTPException(status_code=404, detail='Application not found')

    return {
        'id': str(application['_id']),
        'job': application.get('job', {}),
        'tailored_resume': application.get('tailored_resume', ''),
        'cover_letter': application.get('cover_letter', ''),
        'status': application.get('status', 'Resume Ready'),
        'created_at': application.get('created_at'),
        'updated_at': application.get('updated_at'),
    }


@app.patch('/applications/{application_id}/status')
def update_application_status(
    application_id: str,
    request: ApplicationStatusUpdate,
    user=Depends(get_current_user),
):
    allowed = {
        'Saved', 'Shortlisted', 'Resume Ready', 'Applied',
        'Interview', 'Rejected', 'Offer'
    }
    if request.status not in allowed:
        raise HTTPException(status_code=400, detail='Invalid application status')

    try:
        object_id = ObjectId(application_id)
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid application id')

    result = applications_collection.update_one(
        {'_id': object_id, 'user_id': user['_id']},
        {'$set': {'status': request.status, 'updated_at': now()}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail='Application not found')

    return {'message': 'Application status updated'}


@app.get('/dashboard')
def dashboard(user=Depends(get_current_user)):
    resume = resumes_collection.find_one({'user_id': user['_id']})
    saved_count = saved_jobs_collection.count_documents({'user_id': user['_id']})
    application_count = applications_collection.count_documents({'user_id': user['_id']})
    interview_count = applications_collection.count_documents({
        'user_id': user['_id'],
        'status': 'Interview',
    })

    recent = list(
        applications_collection.find({'user_id': user['_id']})
        .sort('created_at', -1)
        .limit(5)
    )

    status_counts = {}
    for item in applications_collection.find({'user_id': user['_id']}):
        status = item.get('status', 'Resume Ready')
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        'resume': {
            'exists': resume is not None,
            'filename': resume.get('filename', '') if resume else '',
            'updated_at': resume.get('updated_at') if resume else None,
        },
        'stats': {
            'saved_jobs': saved_count,
            'applications': application_count,
            'interviews': interview_count,
        },
        'application_status_counts': status_counts,
        'recent_applications': [
            {
                'id': str(item['_id']),
                'job': item.get('job', {}),
                'status': item.get('status', 'Resume Ready'),
                'created_at': item.get('created_at'),
            }
            for item in recent
        ],
    }
