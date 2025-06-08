from langgraph.graph import StateGraph, END
from .agent import AgentCore

class AgentGraph:
    """LangGraph workflow for the agent."""

    def __init__(self, agent_core:AgentCore, state_type):
        self.agent_core = agent_core
        self.state_type = state_type

    def create_graph(self):
        graph = StateGraph(self.state_type)
        graph.add_node("process_input", self.agent_core.process_input)
        graph.add_node("retrieve_context", self.agent_core.retrieve_context)
        graph.add_node("react_loop", self.agent_core.llm_decide_and_act)
        graph.add_node("update_memory", self.agent_core.update_memory)
        # ReAct loop: LLM decides to use tool or answer, can loop until done
        graph.add_edge("process_input", "retrieve_context")
        graph.add_edge("retrieve_context", "react_loop")
        graph.add_conditional_edges(
            "react_loop",
            lambda state: END if not (getattr(state, '_react_continue', False) if hasattr(state, '__dict__') else state.get('_react_continue')) else "react_loop"
        )
        graph.add_edge("react_loop", "update_memory")
        graph.add_edge("update_memory", END)
        graph.set_entry_point("process_input")
        return graph.compile()