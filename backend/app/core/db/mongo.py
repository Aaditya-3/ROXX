"""
MongoDB connection helper.

Provides a single client and database handle for the application.
"""

import os
from functools import lru_cache
from typing import Any

from pymongo import MongoClient


@lru_cache(maxsize=1)
def get_mongo_client() -> MongoClient:
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    return MongoClient(uri)


@lru_cache(maxsize=1)
def get_db() -> Any:
    client = get_mongo_client()
    db_name = os.getenv("MONGO_DB_NAME", "ai_chat_memory")
    return client[db_name]

