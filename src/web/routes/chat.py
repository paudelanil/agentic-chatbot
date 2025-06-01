from flask import Blueprint, Response, request, jsonify, stream_with_context
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


@chat_bp.route("/chat/stream", methods=["POST"])
@get_or_create_user_session()
def chat_stream():
    data = request.json
    user_id = request.user_id
    if isinstance(user_id, list):
        user_id = user_id[0] if user_id else ""
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    def generate():
        try:
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

            last_content = ""
            final_state = None
            
            # Send initial SSE to establish connection
            yield "data: \n\n"
            
            for state in graph.stream(user_state):
                # Extract the inner state from the node key
                if isinstance(state, dict) and len(state) == 1:
                    inner_state = list(state.values())[0]
                else:
                    inner_state = state
                    
                final_state = inner_state
                messages = inner_state.get("messages", [])
                
                if messages and messages[-1]["role"] == "assistant":
                    content = messages[-1]["content"]
                    if content != last_content:
                        new_text = content[len(last_content):]
                        if new_text.strip():  # Only send non-empty content
                            # Properly escape and format for SSE
                            escaped_text = new_text.replace('\n', '\\n').replace('\r', '\\r')
                            yield f"data: {escaped_text}\n\n"
                        last_content = content
            
            # Save memories after streaming completes
            if final_state:
                agent_core.memory_manager.add_to_memory(user_message, True, user_id)
                if final_state.get("messages") and final_state["messages"][-1]["role"] == "assistant":
                    agent_core.memory_manager.add_to_memory(final_state["messages"][-1]["content"], False, user_id)
            
            # Send end-of-stream signal
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            # Send error message in SSE format
            yield f"data: [ERROR: {str(e)}]\n\n"

    response = Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',  # Disable nginx buffering
        }
    )
    return response