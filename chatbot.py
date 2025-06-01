# chatbot.py
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from flask import Flask, render_template, request, jsonify
import uuid
from typing_extensions import TypedDict
import sqlite3
from functools import wraps
from io import BytesIO

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# PDF processing
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: PyPDF2 not installed. PDF support disabled.")


GROQ_API_KEY =  os.environ.get("GROQ_API_KEY") # This is the default and can be omitted
  # Replace with your actual API key
# MODEL_NAME = "mixtral-8x7b-32768"
MODEL_NAME = 'llama3-8b-8192'

@dataclass
class MemoryItem:
    id: str
    content: str
    timestamp: str
    is_user: bool
    importance: float
    embedding: Optional[List[float]] = None

# Use TypedDict for LangGraph state compatibility
class ChatState(TypedDict):
    messages: List[Dict[str, Any]]
    short_term_memory: List[Dict[str, Any]]
    long_term_memory: List[Dict[str, Any]]
    uploaded_files: List[Dict[str, Any]]
    current_query: str
    tool_results: List[str]
    context: str
    user_id: str  # ADD THIS LINE!


class DatabaseManager:
    def __init__(self, db_path="chatbot.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Memory table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                content TEXT,
                is_user BOOLEAN,
                importance FLOAT,
                embedding BLOB,
                memory_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Files table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_files (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                filename TEXT,
                content TEXT,
                chunks TEXT,
                embeddings BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_memories_user_type 
            ON memories (user_id, memory_type)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_files_user 
            ON user_files (user_id)
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, user_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR IGNORE INTO users (id) VALUES (?)
        ''', (user_id,))
        
        conn.commit()
        conn.close()
    
    def save_memory(self, user_id, memory_item, memory_type):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Ensure user exists
        self.create_user(user_id)
        
        cursor.execute('''
            INSERT OR REPLACE INTO memories 
            (id, user_id, content, is_user, importance, embedding, memory_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory_item['id'], user_id, memory_item['content'],
            memory_item['is_user'], memory_item['importance'],
            json.dumps(memory_item.get('embedding', [])), memory_type
        ))
        
        conn.commit()
        conn.close()
        
        # Clean up old memories to maintain limits
        self.cleanup_old_memories(user_id, memory_type)
    
    def cleanup_old_memories(self, user_id, memory_type):
        """Keep only the most recent memories within limits"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        limit = 100 if memory_type == 'long_term' else 50  # Keep more long-term memories
        
        cursor.execute('''
            DELETE FROM memories 
            WHERE user_id = ? AND memory_type = ? 
            AND id NOT IN (
                SELECT id FROM memories 
                WHERE user_id = ? AND memory_type = ?
                ORDER BY created_at DESC LIMIT ?
            )
        ''', (user_id, memory_type, user_id, memory_type, limit))
        
        conn.commit()
        conn.close()
    
    def load_memories(self, user_id, memory_type, limit=50):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, content, is_user, importance, embedding, created_at FROM memories 
            WHERE user_id = ? AND memory_type = ?
            ORDER BY created_at DESC LIMIT ?
        ''', (user_id, memory_type, limit))
        
        memories = []
        for row in cursor.fetchall():
            memory = {
                'id': row[0],
                'content': row[1],
                'is_user': bool(row[2]),
                'importance': row[3],
                'embedding': json.loads(row[4]) if row[4] else None,
                'timestamp': row[5]
            }
            memories.append(memory)
        
        conn.close()
        return memories
    
    def save_file(self, user_id, file_data):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Ensure user exists
        self.create_user(user_id)
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_files 
            (id, user_id, filename, content, chunks, embeddings)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            file_data['id'], user_id, file_data['filename'],
            file_data['content'], json.dumps(file_data['chunks']),
            json.dumps(file_data['embeddings'])
        ))
        
        conn.commit()
        conn.close()
    
    def load_files(self, user_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, content, chunks, embeddings, created_at FROM user_files 
            WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
        
        files = []
        for row in cursor.fetchall():
            file_data = {
                'id': row[0],
                'filename': row[1],
                'content': row[2],
                'chunks': json.loads(row[3]),
                'embeddings': json.loads(row[4]),
                'timestamp': row[5]
            }
            files.append(file_data)
        
        conn.close()
        return files

class MemoryManager:
    def __init__(self, db_manager):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = db_manager
    
    def calculate_importance(self, message: str) -> float:
        """Calculate importance score for memory storage"""
        important_keywords = ['remember', 'important', 'note', 'save', 'crucial', 'key', 'always']
        score = 5.0  # Base score
        
        # Keyword importance
        for keyword in important_keywords:
            if keyword in message.lower():
                score += 2.0
        
        # Length factor
        if len(message) > 100:
            score += 1.0
        
        # Question factor
        if '?' in message:
            score += 1.0
        
        return min(score, 10.0)
    
    def add_to_memory(self, content: str, is_user: bool, user_id: str):
        """Add item to user's memory in database"""
        memory_item = {
            'id': str(uuid.uuid4()),
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'is_user': is_user,
            'importance': self.calculate_importance(content),
            'embedding': self.embeddings.embed_query(content)
        }
        
        # Save to short-term memory
        self.db.save_memory(user_id, memory_item, 'short_term')
        
        # Save to long-term if important
        if memory_item['importance'] > 7.0:
            self.db.save_memory(user_id, memory_item, 'long_term')
        
        return memory_item
    
    def retrieve_relevant_memories(self, query: str, user_id: str, top_k: int = 5):
        """Retrieve relevant memories using semantic similarity"""
        # Load both types of memories
        short_term = self.db.load_memories(user_id, 'short_term', 20)
        long_term = self.db.load_memories(user_id, 'long_term', 50)
        
        
        all_memories = short_term + long_term
        if not all_memories:
            return []
        
        query_embedding = self.embeddings.embed_query(query)
        similarities = []
        
        for memory in all_memories:
            if memory.get('embedding'):
                similarity = cosine_similarity([query_embedding], [memory['embedding']])[0][0]
                similarities.append((memory, similarity))
        print("Similarities calculated:", similarities)
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in similarities[:top_k]]


class RAGManager:
    def __init__(self): 
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    def process_file(self, file_content: str, filename: str) -> Dict[str, Any]:
        """Process uploaded file for RAG"""
        # Simple text chunking
        chunks = [file_content[i:i+1000] for i in range(0, len(file_content), 800)]  # 200 char overlap
        
        file_data = {
            'id': str(uuid.uuid4()),
            'filename': filename,
            'content': file_content,
            'chunks': chunks,
            'embeddings': [self.embeddings.embed_query(chunk) for chunk in chunks],
            'timestamp': datetime.now().isoformat()
        }
        return file_data
    
    def retrieve_relevant_chunks(self, query: str, files: List[Dict[str, Any]], top_k: int = 3) -> List[str]:
        """Retrieve relevant file chunks for the query"""
        if not files:
            return []
        
        query_embedding = self.embeddings.embed_query(query)
        all_chunks = []
        
        for file in files:
            for i, chunk in enumerate(file['chunks']):
                similarity = cosine_similarity([query_embedding], [file['embeddings'][i]])[0][0]
                all_chunks.append((chunk, similarity, file['filename']))
        
        # Sort by similarity and return top chunks
        all_chunks.sort(key=lambda x: x[1], reverse=True)
        return [f"From {filename}: {chunk}" for chunk, _, filename in all_chunks[:top_k]]

# Tools
@tool
def calculator_tool(expression: str) -> str:
    """Perform mathematical calculations"""
    try:
        # Safe evaluation for basic math
        allowed_chars = set('0123456789+-*/.() ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"Calculation result: {result}"
        else:
            return "Invalid mathematical expression"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

@tool
def datetime_tool() -> str:
    """Get current date and time"""
    return f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def weather_tool(location: str = "current location") -> str:
    """Get weather information (simulated)"""
    import random
    conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "stormy"]
    temp = random.randint(15, 35)
    condition = random.choice(conditions)
    return f"Weather in {location}: {temp}°C, {condition}"

class ChatbotAgent:
    def __init__(self, db_manager):
        self.llm = ChatGroq(
            model=MODEL_NAME,
            temperature=0.7
        )
        self.db_manager = db_manager
        self.memory_manager = MemoryManager(db_manager)
        self.rag_manager = RAGManager()
        self.tools = [calculator_tool, datetime_tool, weather_tool]
    
    def create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        graph = StateGraph(ChatState)
        
        # Add nodes
        graph.add_node("process_input", self.process_input)
        graph.add_node("retrieve_context", self.retrieve_context)
        graph.add_node("check_tools", self.check_tools)
        graph.add_node("generate_response", self.generate_response)
        graph.add_node("update_memory", self.update_memory)
        
        # Add edges
        graph.add_edge("process_input", "retrieve_context")
        graph.add_edge("retrieve_context", "check_tools")
        graph.add_edge("check_tools", "generate_response")
        graph.add_edge("generate_response", "update_memory")
        graph.add_edge("update_memory", END)
        
        # Set entry point
        graph.set_entry_point("process_input")
        
        return graph.compile()
    
    def process_chat(self, user_state, user_id):
        """Process chat with user-specific state and database persistence"""
        # Update state with current query
        user_state["tool_results"] = []  # Reset tool results
        
        # Store user_id in state for access in graph nodes
        user_state["user_id"] = user_id
        
        # Run the graph with user state
        result = self.create_graph().invoke(user_state)
        # Save new memories to database
        if result["messages"]:
            latest_ai_message = result["messages"][-1]['content']
            # Save both user and AI messages
            self.memory_manager.add_to_memory(user_state["current_query"], True, user_id)
            self.memory_manager.add_to_memory(latest_ai_message, False, user_id)
        
        # Get updated memory stats
        short_term_count = len(self.db_manager.load_memories(user_id, 'short_term', 1000))
        long_term_count = len(self.db_manager.load_memories(user_id, 'long_term', 1000))
        
        return {
            'response': result["messages"][-1]['content'] if result["messages"] else "I'm sorry, I couldn't process your request.",
            'memory_stats': {
                'short_term': short_term_count,
                'long_term': long_term_count
            }
        }
    
    def process_input(self, state: ChatState) -> ChatState:
        """Process the input query"""
        # Add user message to conversation
        user_message = {
            "role": "user",
            "content": state["current_query"],
            "timestamp": datetime.now().isoformat()
        }
        state["messages"].append(user_message)
        return state
    
    def retrieve_context(self, state: ChatState) -> ChatState:
        """Retrieve relevant context from memory and files"""
        context_parts = []
        user_id = state.get("user_id")
        print("USER ID", user_id)
        if user_id:
            # Get relevant memories using user_id
            relevant_memories = self.memory_manager.retrieve_relevant_memories(
                state["current_query"], user_id
            )

            if relevant_memories:
                memory_context = "Relevant conversation history:\n"
                for mem in relevant_memories:
                    memory_context += f"- {mem['content']}\n"
                context_parts.append(memory_context)
        
        print("Retrieved context parts:", context_parts)
        # Get relevant file chunks (files are already user-specific in state)
        relevant_chunks = self.rag_manager.retrieve_relevant_chunks(
            state["current_query"], state["uploaded_files"]
        )
        if relevant_chunks:
            file_context = "Relevant information from uploaded files:\n"
            for chunk in relevant_chunks:
                file_context += f"- {chunk}\n"
            context_parts.append(file_context)
        
        state["context"] = "\n\n".join(context_parts)
        return state
    
    def check_tools(self, state: ChatState) -> ChatState:
        """Check if tools should be called"""
        query_lower = state["current_query"].lower()
        
        # Check for calculator
        if any(word in query_lower for word in ['calculate', 'math', '+', '-', '*', '/']):
            # Extract mathematical expression
            import re
            math_pattern = r'[\d+\-*/.() ]+'
            matches = re.findall(math_pattern, state["current_query"])
            if matches:
                result = calculator_tool.invoke({"expression": matches[0]})
                state["tool_results"].append(result)
        
        # Check for datetime
        if any(word in query_lower for word in ['time', 'date', 'today', 'now']):
            result = datetime_tool.invoke({})
            state["tool_results"].append(result)
        
        # Check for weather
        if 'weather' in query_lower:
            # Try to extract location
            import re
            location_match = re.search(r'weather in (.+)', state["current_query"], re.IGNORECASE)
            location = location_match.group(1) if location_match else "current location"
            result = weather_tool.invoke({"location": location})
            state["tool_results"].append(result)
        
        return state
    
    def generate_response(self, state: ChatState) -> ChatState:
        """Generate AI response using LLM"""
        # Prepare system message
        system_prompt = """You are a helpful AI assistant with access to conversation memory and uploaded files. 
        Provide helpful, accurate, and contextual responses. If you used tools or have relevant context, incorporate that information naturally."""
        
        # Prepare messages for LLM
        messages = [SystemMessage(content=system_prompt)]
        
        # Add context if available
        if state["context"]:
            messages.append(SystemMessage(content=f"Relevant context:\n{state['context']}"))
        
        # Add tool results if available
        if state["tool_results"]:
            tool_context = "Tool results:\n" + "\n".join(state["tool_results"])
            messages.append(SystemMessage(content=tool_context))
        
        # Add recent conversation history
        for msg in state["messages"][-5:]:  # Last 5 messages
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        # Generate response
        response = self.llm.invoke(messages)
        
        # Add AI response to conversation
        ai_message = {
            "role": "assistant",
            "content": response.content,
            "timestamp": datetime.now().isoformat()
        }
        state["messages"].append(ai_message)
        
        return state
    
    def update_memory(self, state: ChatState) -> ChatState:
        """Memory updates are handled in process_chat method"""
        return state

# Flask Web Interface
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Session management decorator
def get_or_create_user_session():
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user_id = request.headers.get('X-User-ID') or request.cookies.get('user_id')
            if not user_id:
                user_id = 'user_' + str(uuid.uuid4())
            request.user_id = user_id
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Initialize database and chatbot
db_manager = DatabaseManager()
chatbot = ChatbotAgent(db_manager)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
@get_or_create_user_session()
def chat():
    data = request.json
    user_message = data.get('message', '')
    user_id = request.user_id
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Load user's persistent data
        user_state = {
            "messages": [],
            "short_term_memory": db_manager.load_memories(user_id, 'short_term', 20),
            "long_term_memory": db_manager.load_memories(user_id, 'long_term', 50),
            "uploaded_files": db_manager.load_files(user_id),
            "current_query": user_message,
            "tool_results": [],
            "context": ""
        }

   
        
        # Process with chatbot
        result = chatbot.process_chat(user_state, user_id)
        
        response = jsonify({
            'response': result['response'],
            'memory_stats': result['memory_stats'],
            'files_count': len(user_state["uploaded_files"])
        })
        
        # Set user cookie if new user
        response.set_cookie('user_id', user_id, max_age=30*24*60*60)  # 30 days
        return response
        
    except Exception as e:
        return jsonify({'error': f'Error processing message: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
@get_or_create_user_session()
def upload_file():
    user_id = request.user_id
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Check file type and process accordingly
        if file.filename.lower().endswith('.pdf') and PDF_SUPPORT:
            # Process PDF file
            pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
            file_content = ""
            for page in pdf_reader.pages:
                file_content += page.extract_text() + "\n"
        elif file.filename.lower().endswith('.pdf') and not PDF_SUPPORT:
            return jsonify({'error': 'PDF support not available. Install PyPDF2 to enable PDF uploads.'}), 400
        else:
            # Process text files
            file_content = file.read().decode('utf-8')
        
        # Process file for RAG
        file_data = chatbot.rag_manager.process_file(file_content, file.filename)
        
        # Save to database
        db_manager.save_file(user_id, file_data)
        
        response = jsonify({
            'message': f'File {file.filename} uploaded successfully',
            'files_count': len(db_manager.load_files(user_id))
        })
        
        # Set user cookie
        response.set_cookie('user_id', user_id, max_age=30*24*60*60)  # 30 days
        return response
    
    except Exception as e:
        return jsonify({'error': f'Error uploading file: {str(e)}'}), 500

@app.route('/memory')
@get_or_create_user_session()
def get_memory():
    user_id = request.user_id
    
    short_term = db_manager.load_memories(user_id, 'short_term', 10)
    long_term = db_manager.load_memories(user_id, 'long_term', 20)
    
    response = jsonify({
        'short_term': short_term,
        'long_term': long_term,
        'stats': {
            'short_term_count': len(db_manager.load_memories(user_id, 'short_term', 1000)),
            'long_term_count': len(db_manager.load_memories(user_id, 'long_term', 1000))
        }
    })
    
    # Set user cookie
    response.set_cookie('user_id', user_id, max_age=30*24*60*60)  # 30 days
    return response

@app.route('/files')
@get_or_create_user_session()
def get_files():
    user_id = request.user_id
    files = db_manager.load_files(user_id)
    
    files_info = []
    for file in files:
        files_info.append({
            'id': file['id'],
            'filename': file['filename'],
            'timestamp': file['timestamp'],
            'chunks_count': len(file['chunks'])
        })
    
    response = jsonify({'files': files_info})
    
    # Set user cookie
    response.set_cookie('user_id', user_id, max_age=30*24*60*60)  # 30 days
    return response

@app.route('/clear_memory', methods=['POST'])
@get_or_create_user_session()
def clear_memory():
    """Clear user's memory (for testing/reset purposes)"""
    user_id = request.user_id
    
    try:
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()
        
        # Delete user's memories
        cursor.execute('DELETE FROM memories WHERE user_id = ?', (user_id,))
        
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'Memory cleared successfully'})
    
    except Exception as e:
        return jsonify({'error': f'Error clearing memory: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting LangGraph Chatbot with Persistent Memory...")


    app.run(debug=True, host='0.0.0.0', port=8000)