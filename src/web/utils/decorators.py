from functools import wraps
from flask import request
import uuid

def get_or_create_user_session():
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user_id = request.headers.get("X-User-ID") or request.cookies.get("user_id")
            if not user_id:
                user_id = "user_" + str(uuid.uuid4())
            request.user_id = user_id
            return f(*args, **kwargs)
        return decorated_function
    return decorator