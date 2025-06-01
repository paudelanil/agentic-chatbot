"""
Calculator tool for mathematical operations.
"""
import logging
import re
import ast
import operator
from langchain.tools import tool

logger = logging.getLogger(__name__)

# Safe mathematical operations
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Safe mathematical functions
SAFE_FUNCTIONS = {
    'abs': abs,
    'round': round,
    'min': min,
    'max': max,
    'sum': sum,
}

class SafeMathEvaluator(ast.NodeVisitor):
    """Safe evaluator for mathematical expressions."""
    
    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = SAFE_OPERATORS.get(type(node.op))
        
        if op is None:
            raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
        
        try:
            return op(left, right)
        except ZeroDivisionError:
            raise ValueError("Division by zero")
        except Exception as e:
            raise ValueError(f"Mathematical error: {str(e)}")
    
    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        op = SAFE_OPERATORS.get(type(node.op))
        
        if op is None:
            raise ValueError(f"Unsupported unary operation: {type(node.op).__name__}")
        
        return op(operand)
    
    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed")
        
        func_name = node.func.id
        if func_name not in SAFE_FUNCTIONS:
            raise ValueError(f"Unsupported function: {func_name}")
        
        args = [self.visit(arg) for arg in node.args]
        
        try:
            return SAFE_FUNCTIONS[func_name](*args)
        except Exception as e:
            raise ValueError(f"Function error: {str(e)}")
    
    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")
    
    def visit_Name(self, node):
        # Allow mathematical constants
        constants = {
            'pi': 3.141592653589793,
            'e': 2.718281828459045,
        }
        
        if node.id in constants:
            return constants[node.id]
        
        raise ValueError(f"Undefined variable: {node.id}")
    
    def generic_visit(self, node):
        raise ValueError(f"Unsupported syntax: {type(node).__name__}")

def safe_eval(expression: str) -> float:
    """Safely evaluate a mathematical expression."""
    try:
        # Parse the expression
        tree = ast.parse(expression, mode='eval')
        
        # Evaluate using our safe evaluator
        evaluator = SafeMathEvaluator()
        result = evaluator.visit(tree.body)
        
        return float(result)
    
    except SyntaxError:
        raise ValueError("Invalid mathematical expression syntax")
    except Exception as e:
        raise ValueError(f"Evaluation error: {str(e)}")

@tool
def calculator_tool(expression: str) -> str:
    """
    Perform mathematical calculations safely.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4", "sqrt(16)", "abs(-5)")
    
    Returns:
        String containing the calculation result or error message
    
    Examples:
        - Basic arithmetic: "2 + 3 * 4" -> "14"
        - Functions: "abs(-5)" -> "5"
        - Constants: "pi * 2" -> "6.283185307179586"
    """
    try:
        # Clean the expression
        expression = expression.strip()
        
        if not expression:
            return "Error: Empty expression provided"
        
        # Basic validation - check for allowed characters
        allowed_pattern = r'^[0-9+\-*/().pi e\s,a-z_]+$'
        if not re.match(allowed_pattern, expression, re.IGNORECASE):
            return "Error: Expression contains invalid characters"
        
        # Evaluate the expression
        result = safe_eval(expression)
        
        # Format the result
        if result.is_integer():
            formatted_result = str(int(result))
        else:
            formatted_result = f"{result:.10g}"  # Remove unnecessary trailing zeros
        
        logger.debug(f"Calculator: {expression} = {formatted_result}")
        return f"Calculation result: {formatted_result}"
    
    except ValueError as e:
        error_msg = f"Error: {str(e)}"
        logger.warning(f"Calculator error for '{expression}': {error_msg}")
        return error_msg
    
    except Exception as e:
        error_msg = f"Unexpected error in calculation: {str(e)}"
        logger.error(f"Calculator unexpected error for '{expression}': {error_msg}")
        return error_msg

def extract_math_expressions(text: str) -> list:
    """Extract potential mathematical expressions from text."""
    # Pattern to match mathematical expressions
    patterns = [
        r'\b\d+(?:\.\d+)?\s*[+\-*/]\s*\d+(?:\.\d+)?\b',  # Basic arithmetic
        r'\b(?:sqrt|abs|round|min|max|sum)\s*\([^)]+\)',  # Function calls
        r'\b\d+(?:\.\d+)?\s*[\^]\s*\d+(?:\.\d+)?\b',     # Power operations
    ]
    
    expressions = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        expressions.extend(matches)
    
    return expressions