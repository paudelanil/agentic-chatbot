from langchain.tools import tool

@tool("appointment", return_direct=True)
def appointment_tool(date: str, time: str, description: str) -> str:
    """Book or manage an appointment. Provide date, time, and description. Date and time should be explicit (e.g., '2025-06-08', '15:00'). If the user provides a relative date/time, use the datetime_tool to resolve it first."""
    return f"Appointment booked for {date} at {time}. Details: {description}"
