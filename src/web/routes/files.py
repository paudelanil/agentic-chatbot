from flask import Blueprint, request, jsonify
from src.database.manager import DatabaseManager
from src.core.agent import AgentCore
from src.web.utils.decorators import get_or_create_user_session
from io import BytesIO

db_manager = DatabaseManager()
agent_core = AgentCore(db_manager)

files_bp = Blueprint("files", __name__)

@files_bp.route("/upload", methods=["POST"])
@get_or_create_user_session()
def upload_file():
    user_id = request.user_id
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        if file.filename.lower().endswith(".pdf"):
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
                file_content = "".join(page.extract_text() or "" for page in pdf_reader.pages)
            except ImportError:
                return jsonify({"error": "PDF support not available. Install PyPDF2."}), 400
        else:
            file_content = file.read().decode("utf-8")

        file_data = agent_core.rag_manager.process_file(file_content, file.filename,user_id)
        db_manager.save_file(user_id, file_data)

        return jsonify({
            "message": f"File {file.filename} uploaded successfully",
            "files_count": len(db_manager.load_files(user_id))
        })
    except Exception as e:
        return jsonify({"error": f"Error uploading file: {str(e)}"}), 500

@files_bp.route("/files", methods=["GET"])
@get_or_create_user_session()
def get_files():
    user_id = request.user_id
    files = db_manager.load_files(user_id)
    files_info = [
        {
            "id": f.id,
            "filename": f.filename,
            "timestamp": f.created_at.isoformat(),
            "chunks_count": len(f.chunks)
        }
        for f in files
        ]
    return jsonify({"files": files_info})

@files_bp.route("/remove", methods=["POST"])
@get_or_create_user_session()
def remove_file():
    user_id = request.user_id
    data = request.get_json()
    file_id = data.get("file_id")
    if not file_id:
        return jsonify({"error": "No file_id provided"}), 400
    try:
        removed = db_manager.remove_file(user_id, file_id)
        if not removed:
            return jsonify({"error": "File not found or could not be removed"}), 404
        return jsonify({"message": f"File {file_id} removed successfully"})
    except Exception as e:
        return jsonify({"error": f"Error removing file: {str(e)}"}), 500