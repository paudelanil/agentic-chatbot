
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


# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import PyPDF2
from io import BytesIO

# Configuration
GROQ_API_KEY =  os.environ.get("GROQ_API_KEY"),  # This is the default and can be omitted
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
    short_term_memory: List[Dict[str, Any]]  # Store as dicts for serialization
    long_term_memory: List[Dict[str, Any]]
    uploaded_files: List[Dict[str, Any]]
    current_query: str
    tool_results: List[str]
    context: str

class MemoryManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
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
    
    def add_to_memory(self, content: str, is_user: bool, short_term: List[Dict], long_term: List[Dict]):
        """Add item to appropriate memory stores"""
        memory_item = {
            'id': str(uuid.uuid4()),
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'is_user': is_user,
            'importance': self.calculate_importance(content),
            'embedding': self.embeddings.embed_query(content)
        }
        
        # Add to short-term (keep last 20 items)
        short_term.append(memory_item)
        if len(short_term) > 20:
            short_term.pop(0)
        
        # Add to long-term if important (keep last 100 items)
        if memory_item['importance'] > 7.0:
            long_term.append(memory_item)
            if len(long_term) > 100:
                long_term.pop(0)
        
        return short_term, long_term
    
    def retrieve_relevant_memories(self, query: str, short_term: List[Dict], long_term: List[Dict], top_k: int = 5) -> List[Dict]:
        """Retrieve relevant memories using semantic similarity"""
        all_memories = short_term + long_term
        if not all_memories:
            return []
        
        query_embedding = self.embeddings.embed_query(query)
        similarities = []
        
        for memory in all_memories:
            if memory.get('embedding'):
                similarity = cosine_similarity([query_embedding], [memory['embedding']])[0][0]
                similarities.append((memory, similarity))
        
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
    def __init__(self):
        self.llm = ChatGroq(
            # groq_api_key=GROQ_API_KEY,
            model=MODEL_NAME,
            temperature=0.7
        )
        self.memory_manager = MemoryManager()
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
        
        # Get relevant memories
        relevant_memories = self.memory_manager.retrieve_relevant_memories(
            state["current_query"], state["short_term_memory"], state["long_term_memory"]
        )
        if relevant_memories:
            memory_context = "Relevant conversation history:\n"
            for mem in relevant_memories:
                memory_context += f"- {mem['content']}\n"
            context_parts.append(memory_context)
        
        # Get relevant file chunks
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
        """Update memory with new conversation"""
        # Add user message to memory
        state["short_term_memory"], state["long_term_memory"] = self.memory_manager.add_to_memory(
            state["current_query"], True, state["short_term_memory"], state["long_term_memory"]
        )
        
        # Add AI response to memory
        ai_response = state["messages"][-1]["content"]
        state["short_term_memory"], state["long_term_memory"] = self.memory_manager.add_to_memory(
            ai_response, False, state["short_term_memory"], state["long_term_memory"]
        )
        
        return state

# Flask Web Interface
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Global chatbot instance
chatbot = ChatbotAgent()
graph = chatbot.create_graph()

# Global state (in production, use proper session management)
chat_state = {
    "messages": [],
    "short_term_memory": [],
    "long_term_memory": [],
    "uploaded_files": [],
    "current_query": "",
    "tool_results": [],
    "context": ""
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global chat_state
    
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Update state with current query
        chat_state["current_query"] = user_message
        chat_state["tool_results"] = []  # Reset tool results
        
        # Run the graph
        result = graph.invoke(chat_state)
        
        # Get the latest AI response
        ai_response = result["messages"][-1]['content'] if result["messages"] else "I'm sorry, I couldn't process your request."
        
        # Update global state
        chat_state.update(result)
        
        return jsonify({
            'response': ai_response,
            'memory_stats': {
                'short_term': len(chat_state["short_term_memory"]),
                'long_term': len(chat_state["long_term_memory"])
            },
            'files_count': len(chat_state["uploaded_files"])
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing message: {str(e)}'}), 500

# Update the upload route:
@app.route('/upload', methods=['POST'])
def upload_file():
    global chat_state
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Check file type and process accordingly
        if file.filename.lower().endswith('.pdf'):
            # Process PDF file
            pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
            file_content = ""
            for page in pdf_reader.pages:
                file_content += page.extract_text() + "\n"
        else:
            # Process text files
            file_content = file.read().decode('utf-8')
        
        # Process file for RAG
        file_data = chatbot.rag_manager.process_file(file_content, file.filename)
        chat_state["uploaded_files"].append(file_data)
        
        return jsonify({
            'message': f'File {file.filename} uploaded successfully',
            'files_count': len(chat_state["uploaded_files"])
        })
    
    except Exception as e:
        return jsonify({'error': f'Error uploading file: {str(e)}'}), 500

@app.route('/memory')
def get_memory():
    global chat_state

    return jsonify({
        'short_term': chat_state["short_term_memory"][-10:],
        'long_term': chat_state["long_term_memory"][-20:],
        'stats': {
            'short_term_count': len(chat_state['short_term_memory']),
            'long_term_count': len(chat_state['long_term_memory'])
        }
    })

@app.route('/files')
def get_files():
    global chat_state
    
    files_info = []
    for file in chat_state['uploaded_files']:
        files_info.append({
            'id': file['id'],
            'filename': file['filename'],
            'timestamp': file['timestamp'],
            'chunks_count': len(file['chunks'])
        })
    
    return jsonify({'files': files_info})

if __name__ == '__main__':
    print("Starting LangGraph Chatbot...")
    app.run(debug=True, host='0.0.0.0', port=8000)
