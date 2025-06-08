from langchain_groq import ChatGroq
from ..memory.manager import MemoryManager
from ..rag.manager import RAGManager
from ..tools.calculator import calculator_tool
from ..tools.datetime_tool import datetime_tool
from ..tools.weather import weather_tool
from ..tools.appointment import appointment_tool
from ..tools import AVAILABLE_TOOLS
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
        self.tools = AVAILABLE_TOOLS
        self.tool_map = {tool.name: tool for tool in self.tools}
        # Bind tools to LLM for ReAct agent
        self.llm_with_tools = self.llm.bind_tools(self.tools)

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
        uploaded_files = state.get("uploaded_files", [])
        # Defensive: if uploaded_files is a list of dicts, extract user_id; if it's a list of lists, flatten or fix
        if uploaded_files and isinstance(uploaded_files, list):
            if all(isinstance(f, dict) for f in uploaded_files):
                # Use user_id as before
                relevant_chunks = self.rag_manager.retrieve_relevant_chunks(
                    state["current_query"], user_id
                )
            else:
                print(f"[DEBUG] uploaded_files is not a list of dicts: {uploaded_files}")
                relevant_chunks = []
        else:
            relevant_chunks = []
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


    def llm_decide_and_act(self, state):
        state = asdict(state)
        from langchain.schema import SystemMessage, HumanMessage, AIMessage
        # User-friendly ReAct-style prompt: do not mention tool names or tool-calling in responses
        system_prompt = (
            "You are a helpful AI assistant. You have access to the following tools: "
            + ", ".join([f"{tool.name} ({tool.description})" for tool in self.tools]) + ". "
            "If the user request is missing information, ask for it conversationally and naturally. "
            "When you have all the information, use the appropriate tool internally, but do not mention tool names or tool usage in your response. "
            "After completing a task, simply provide the result to the user in a friendly, natural way. "
            "If no tool is needed, answer directly."
        )
        messages = [SystemMessage(content=system_prompt)]
        if state.get("context"):
            messages.append(SystemMessage(content=f"Relevant context:\n{state['context']}"))
        for msg in state["messages"][-5:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        response = self.llm_with_tools.invoke(messages)
        # If a tool was used, response will have .tool_calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            for call in response.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]
                tool = self.tool_map.get(tool_name)
                if tool:
                    tool_result = tool.invoke(tool_args)
                    state["messages"].append({
                        "role": "assistant",
                        "content": str(tool_result),
                        "timestamp": state.get("timestamp")
                    })
                    state["tool_results"].append(f"{tool_name}: {tool_result}")
                    state["_react_continue"] = True
                    return state
        # Otherwise, treat as final answer
        ai_message = {
            "role": "assistant",
            "content": response.content.strip(),
            "timestamp": state.get("timestamp")
        }
        state["messages"].append(ai_message)
        state["_react_continue"] = False
        return state