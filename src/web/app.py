from flask import Flask, render_template
import os
from .routes.chat import chat_bp
from .routes.files import files_bp
from .routes.memory import memory_bp
from flask_cors import CORS

# Get the absolute path to the templates directory at the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATES_DIR)
app.secret_key = "your-secret-key-here"

# Disable buffering for better streaming
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Enable CORS with proper streaming headers
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization", "X-User-ID"],
        "expose_headers": ["X-Accel-Buffering"]
    }
})

# Register blueprints
app.register_blueprint(chat_bp)
app.register_blueprint(files_bp)
app.register_blueprint(memory_bp)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000, threaded=True)