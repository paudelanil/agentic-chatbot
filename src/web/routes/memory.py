from flask import Blueprint, jsonify, request
from src.database.manager import DatabaseManager
from src.database.models import MemoryType  # <-- Import the Enum
from src.web.utils.decorators import get_or_create_user_session

db_manager = DatabaseManager()

memory_bp = Blueprint("memory", __name__)

from dataclasses import asdict

@memory_bp.route("/memory", methods=["GET"])
@get_or_create_user_session()
def get_memory():
    user_id = request.user_id
    short_term = db_manager.load_memories(user_id, MemoryType.SHORT_TERM, 10)
    long_term = db_manager.load_memories(user_id, MemoryType.LONG_TERM, 20)

    # Convert dataclass objects to dicts and enums to their values
    def serialize_memory(mem):
        d = asdict(mem)
        d["memory_type"] = mem.memory_type.value  # Enum to string
        d["created_at"] = mem.created_at.isoformat()  # datetime to string
        return d

    short_term_serialized = [serialize_memory(m) for m in short_term]
    long_term_serialized = [serialize_memory(m) for m in long_term]

    return jsonify({
        "short_term": short_term_serialized,
        "long_term": long_term_serialized,
        "stats": {
            "short_term_count": len(db_manager.load_memories(user_id, MemoryType.SHORT_TERM, 10)),
            "long_term_count": len(db_manager.load_memories(user_id, MemoryType.LONG_TERM, 20))
        }
    })

@memory_bp.route("/clear_memory", methods=["POST"])
@get_or_create_user_session()
def clear_memory():
    user_id = request.user_id
    try:
        db_manager.delete_user_memories(user_id)
        return jsonify({"message": "Memory cleared successfully"})
    except Exception as e:
        return jsonify({"error": f"Error clearing memory: {str(e)}"}), 500