from langchain_groq import ChatGroq
from ..memory.manager import MemoryManager
from ..rag.manager import RAGManager
from ..tools.calculator import calculator_tool
from ..tools.datetime_tool import datetime_tool
from ..tools.weather import weather_tool
from config.settings import settings
from dataclasses import asdict

class AgentCore:
    """Core agent logic for processing chat and managing tools/memory."""

    def __init__(self, db_manager):
        self.llm = ChatGroq(
            model=settings.llm.model_name,
            temperature=settings.llm.temperature
        )
        self.memory_manager = MemoryManager(db_manager)
        self.rag_manager = RAGManager(db_manager)
        self.tools = [calculator_tool, datetime_tool, weather_tool]

    def process_input(self, state):
        """Add user message to conversation."""
        state = asdict(state)
        user_message = {
            "role": "user",
            "content": state["current_query"],
            "timestamp": state.get("timestamp")
        }
        state["messages"].append(user_message)
        return state

    def retrieve_context(self, state):
        """Retrieve relevant context from memory and files."""
        state = asdict(state)
        
        user_id = state.get("user_id")
        context_parts = []
        if user_id:
            memory_context = self.memory_manager.get_memory_context(state["current_query"], user_id)
            if memory_context:
                context_parts.append(memory_context)
        # RAG context (if files present)
        relevant_chunks = self.rag_manager.retrieve_relevant_chunks(
            state["current_query"], state.get("uploaded_files", [])
        )
        if relevant_chunks:
            file_context = "Relevant information from uploaded files:\n"
            for chunk in relevant_chunks:
                file_context += f"- {chunk}\n"
            context_parts.append(file_context)
        state["context"] = "\n\n".join(context_parts)
        return state

    def check_tools(self, state):
        state = asdict(state)
        """Check if tools should be called and append results."""
        query_lower = state["current_query"].lower()
        # Calculator
        if any(word in query_lower for word in ['calculate', 'math', '+', '-', '*', '/']):
            import re
            math_pattern = r'[\d+\-*/.() ]+'
            matches = re.findall(math_pattern, state["current_query"])
            if matches:
                result = calculator_tool.invoke({"expression": matches[0]})
                state["tool_results"].append(result)
        # Datetime
        if any(word in query_lower for word in ['time', 'date', 'today', 'now']):
            result = datetime_tool.invoke({})
            state["tool_results"].append(result)
        # Weather
        if 'weather' in query_lower:
            import re
            location_match = re.search(r'weather in (.+)', state["current_query"], re.IGNORECASE)
            location = location_match.group(1) if location_match else "current location"
            result = weather_tool.invoke({"location": location})
            state["tool_results"].append(result)
        return state

    def generate_response(self, state):
        state = asdict(state)
        """Generate AI response using LLM."""
        from langchain.schema import SystemMessage, HumanMessage, AIMessage
        system_prompt = (
            "You are a helpful AI assistant with access to conversation memory and uploaded files. "
            "Provide helpful, accurate, and contextual responses. If you used tools or have relevant context, "
            "incorporate that information naturally."
        )
        messages = [SystemMessage(content=system_prompt)]
        if state.get("context"):
            messages.append(SystemMessage(content=f"Relevant context:\n{state['context']}"))
        if state.get("tool_results"):
            tool_context = "Tool results:\n" + "\n".join(state["tool_results"])
            messages.append(SystemMessage(content=tool_context))
        for msg in state["messages"][-5:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        response = self.llm.invoke(messages)
        ai_message = {
            "role": "assistant",
            "content": response.content,
            "timestamp": state.get("timestamp")
        }
        state["messages"].append(ai_message)
        return state

    def update_memory(self, state):
        """Memory updates are handled outside the graph."""
        return state