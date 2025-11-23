"""Format LLM logs from JSON output into readable text.

Converts escaped newlines to actual newlines and wraps long lines for readability.
"""
import json
import sys
import textwrap
from pathlib import Path


def format_text(text: str, width: int = 100) -> str:
    """Format text with proper newlines and line wrapping.
    
    Args:
        text: Text to format (may contain \n escape sequences)
        width: Maximum line width for wrapping
    
    Returns:
        Formatted text with actual newlines
    """
    if not text:
        return ""
    
    # Replace escaped newlines with actual newlines
    text = text.replace('\\n', '\n')
    
    # Split by existing newlines
    lines = text.split('\n')
    
    # Wrap each line if it's too long
    wrapped_lines = []
    for line in lines:
        if len(line) <= width:
            wrapped_lines.append(line)
        else:
            # Wrap long lines, preserving indentation
            indent = len(line) - len(line.lstrip())
            wrapped = textwrap.fill(
                line,
                width=width,
                subsequent_indent=' ' * indent,
                break_long_words=False,
                break_on_hyphens=False
            )
            wrapped_lines.append(wrapped)
    
    return '\n'.join(wrapped_lines)


def format_llm_call(call: dict, index: int) -> str:
    """Format a single LLM call into readable text.
    
    Args:
        call: Dictionary containing LLM call info
        index: Index of this call
    
    Returns:
        Formatted string representation
    """
    output = []
    output.append("=" * 100)
    output.append(f"LLM CALL #{index + 1}")
    output.append("=" * 100)
    output.append("")
    
    # Format request
    output.append("REQUEST:")
    output.append("-" * 100)
    
    # request is a list of message dicts
    messages = call.get("request", [])
    
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        output.append(f"\n[{role.upper()}]")
        output.append(format_text(content))
    
    
    # Format response
    output.append("\n")
    output.append("RESPONSE:")
    output.append("-" * 100)
    response = call.get("response", "")
    output.append(format_text(response))
    
    output.append("\n")
    return "\n".join(output)


def format_llm_logs(json_file: str, output_file: str = None):
    """Format LLM logs from JSON file.
    
    Args:
        json_file: Path to JSON file containing LLM logs
        output_file: Optional path to save formatted output (default: prints to stdout)
    """
    # Load JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get LLM calls
    llm_calls = data.get("llm_calls", [])
    
    if not llm_calls:
        print("No LLM calls found in the JSON file.")
        return
    
    # Format all calls
    formatted_output = []
    formatted_output.append(f"TOTAL LLM CALLS: {len(llm_calls)}")
    formatted_output.append("")
    
    for i, call in enumerate(llm_calls):
        formatted_output.append(format_llm_call(call, i))
    
    result = "\n".join(formatted_output)
    
    # Output
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"Formatted logs saved to: {output_file}")
    else:
        print(result)