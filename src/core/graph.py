from langgraph.graph import StateGraph, END

class AgentGraph:
    """LangGraph workflow for the agent."""

    def __init__(self, agent_core, state_type):
        self.agent_core = agent_core
        self.state_type = state_type

    def create_graph(self):
        graph = StateGraph(self.state_type)
        graph.add_node("process_input", self.agent_core.process_input)
        graph.add_node("retrieve_context", self.agent_core.retrieve_context)
        graph.add_node("check_tools", self.agent_core.check_tools)
        graph.add_node("generate_response", self.agent_core.generate_response)
        graph.add_node("update_memory", self.agent_core.update_memory)
        graph.add_edge("process_input", "retrieve_context")
        graph.add_edge("retrieve_context", "check_tools")
        graph.add_edge("check_tools", "generate_response")
        graph.add_edge("generate_response", "update_memory")
        graph.add_edge("update_memory", END)
        graph.set_entry_point("process_input")
        return graph.compile()