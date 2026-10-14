import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi

load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'job_search_system')

if not MONGODB_URI:
    raise RuntimeError('MONGODB_URI is missing from .env')

client = MongoClient(
    MONGODB_URI,
    server_api=ServerApi('1', strict=True, deprecation_errors=True)
)

db = client[MONGODB_DB_NAME]

users_collection = db['users']
resumes_collection = db['resumes']
saved_jobs_collection = db['saved_jobs']
applications_collection = db['applications']
workflow_sessions_collection = db['workflow_sessions']

users_collection.create_index('email', unique=True)
saved_jobs_collection.create_index([('user_id', 1), ('job_id', 1)], unique=True)
applications_collection.create_index([('user_id', 1), ('created_at', -1)])
workflow_sessions_collection.create_index('thread_id', unique=True)


def check_connection():
    client.admin.command({'ping': 1})
    return True
