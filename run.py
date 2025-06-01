from src.web.app import app
import sys

if __name__ == "__main__":
    # Disable stdout buffering for better streaming
    sys.stdout.reconfigure(line_buffering=True)
    app.run(debug=True, host="0.0.0.0", port=8000, threaded=True)   