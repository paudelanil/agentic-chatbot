from langchain.tools import tool
@tool
def weather_tool(location: str = "current location") -> str:
    """Get weather information (simulated)"""
    import random
    conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "stormy"]
    temp = random.randint(15, 35)
    condition = random.choice(conditions)
    return f"Weather in {location}: {temp}°C, {condition}"