# Agentic Chatbot

A modular, extensible chatbot framework built with Python, designed for rapid prototyping and deployment of agent-based conversational AI systems. Supports RAG (Retrieval-Augmented Generation), memory, tool use, and web API integration.

## Features
- **Agentic architecture**: Easily add, modify, or extend agents and tools.
- **RAG support**: Integrate external knowledge sources for more accurate responses.
- **Memory management**: Store and retrieve conversation history and context.
- **Tool integrations**: Calculator, weather, appointment, GitHub MCP, and more.
- **Web API**: RESTful endpoints for chat, file upload, and memory management.
- **Dockerized**: Easy deployment with Docker and Docker Compose.

## How the Project is Structured

The project is organized for modularity and clarity:

- **chatbot.py**: Main chatbot logic and orchestration.
- **run.py**: Entry point to start the FastAPI web server.
- **requirements.txt / pyproject.toml**: Python dependencies and project metadata.
- **Dockerfile / docker-compose.yml**: Containerization and deployment setup.
- **chroma_db/**: Vector database files for RAG (Retrieval-Augmented Generation).
- **config/**: Configuration files and environment settings.
- **src/**: Main source code directory:
  - **core/**: Agent and graph logic (core agentic behavior).
  - **database/**: Database models and manager for persistent storage.
  - **memory/**: Memory manager for conversation context.
  - **rag/**: RAG manager for retrieval-augmented generation.
  - **tools/**: Integrations for tools (calculator, weather, GitHub MCP, etc.).
  - **utils/**: Utility functions and helpers.
  - **web/**: FastAPI web app, including routes and web utilities.
    - **routes/**: API endpoints for chat, file upload, and memory.
    - **utils/**: Web-specific utilities and decorators.
- **static/**: Static files (HTML, CSS, JS) for the web interface.
- **templates/**: Jinja2 templates for rendering web pages.
- **tests/**: Unit and integration tests.
- **uploads/**: Directory for uploaded files.

This structure allows for easy extension, testing, and deployment of new features and agents.

## Getting Started

### Prerequisites
- Python 3.11+
- (Optional) Docker & Docker Compose

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/paudelanil/agentic-chatbot.git
   cd agentic-chatbot
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. (Optional) Set up environment variables in `config/settings.py` as needed.

### Running the App
#### Locally
```sh
python run.py
```
The API will be available at `http://localhost:8000`.

#### With Docker
```sh
docker-compose up --build
```

### API Endpoints
- `/chat` — Chat with the agent (POST)
- `/files` — Upload files (POST)
- `/memory` — Manage conversation memory

See `src/web/routes/` for more endpoints and details.

## Testing
Run all tests with:
```sh
pytest tests/
```

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
MIT License

## Author
[Anil Paudel](https://github.com/paudelanil)
