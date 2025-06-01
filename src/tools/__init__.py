"""
Tools module for the chatbot.
This module provides various tools that the chatbot can use to perform specific tasks.
"""
from .calculator import calculator_tool
from .datetime_tool import datetime_tool
from .weather import weather_tool

# Registry of all available tools
AVAILABLE_TOOLS = [
    calculator_tool,
    datetime_tool,
    weather_tool,
]

# Tool categories for organization
TOOL_CATEGORIES = {
    'math': [calculator_tool],
    'utility': [datetime_tool],
    'information': [weather_tool],
}

def get_all_tools():
    """Get all available tools."""
    return AVAILABLE_TOOLS

def get_tools_by_category(category: str):
    """Get tools by category."""
    return TOOL_CATEGORIES.get(category, [])

def get_tool_by_name(name: str):
    """Get a specific tool by name."""
    for tool in AVAILABLE_TOOLS:
        if tool.name == name:
            return tool
    return None

__all__ = [
    'calculator_tool',
    'datetime_tool', 
    'weather_tool',
    'AVAILABLE_TOOLS',
    'TOOL_CATEGORIES',
    'get_all_tools',
    'get_tools_by_category',
    'get_tool_by_name',
]