from flask import Blueprint, request, jsonify
from ...core.agent import AgentCore
from ...core.graph import AgentGraph
from ...database.manager import DatabaseManager
from ...database.models import ChatState
from ..utils.decorators import get_or_create_user_session

db_manager = DatabaseManager()
agent_core = AgentCore(db_manager)
agent_graph = AgentGraph(agent_core, ChatState)
graph = agent_graph.create_graph()

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/chat", methods=["POST"])
@get_or_create_user_session()
def chat():
    data = request.json
    user_id = request.user_id
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    user_state = {
        "messages": data.get("messages", []),
        "short_term_memory": db_manager.load_memories(user_id, "short_term", 20),
        "long_term_memory": db_manager.load_memories(user_id, "long_term", 50),
        "uploaded_files": db_manager.load_files(user_id),
        "current_query": user_message,
        "tool_results": [],
        "context": "",
        "user_id": user_id,
        "timestamp": data.get("timestamp"),
    }

    result_state = graph.invoke(user_state)

    # Save new memories
    agent_core.memory_manager.add_to_memory(user_message, True, user_id)
    if result_state["messages"]:
        agent_core.memory_manager.add_to_memory(result_state["messages"][-1]["content"], False, user_id)

    return jsonify({
        "response": result_state["messages"][-1]["content"] if result_state["messages"] else "Sorry, no response.",
        "messages": result_state["messages"]
    })