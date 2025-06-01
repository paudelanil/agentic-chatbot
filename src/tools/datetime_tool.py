"""
DateTime tool for date and time operations.
"""
import logging
from datetime import datetime, timedelta, timezone
from langchain.tools import tool

logger = logging.getLogger(__name__)

@tool
def datetime_tool(format_type: str = "standard") -> str:
    """
    Get current date and time information.
    
    Args:
        format_type: Type of format to return
            - "standard": Standard datetime format (default)
            - "date": Date only
            - "time": Time only  
            - "iso": ISO format
            - "timestamp": Unix timestamp
            - "relative": Relative format (day of week, etc.)
    
    Returns:
        String containing the requested date/time information
    """
    try:
        now = datetime.now()
        
        if format_type == "date":
            result = now.strftime('%Y-%m-%d')
        elif format_type == "time":
            result = now.strftime('%H:%M:%S')
        elif format_type == "iso":
            result = now.isoformat()
        elif format_type == "timestamp":
            result = str(int(now.timestamp()))
        elif format_type == "relative":
            day_name = now.strftime('%A')
            month_name = now.strftime('%B')
            result = f"{day_name}, {month_name} {now.day}, {now.year} at {now.strftime('%H:%M:%S')}"
        else:  # Default to standard format
            result = now.strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Current datetime ({format_type}): {result}")
        return result
    except Exception as e:
        logger.error(f"Error getting datetime: {e}")
        return f"Error: {str(e)}"
