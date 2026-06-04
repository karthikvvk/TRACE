import os
import sys
import argparse
import asyncio
from typing import Any, List, Dict
import json

from dotenv import load_dotenv
load_dotenv()  # Load GEMINI_API_KEY and other vars from .env


# Add parent directory to sys.path so we can import 'backend'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.engine.router import ToolRouter

# Initialize ToolRouter
router = ToolRouter()


def clean_schema(schema: dict, for_gemini: bool = False) -> dict:
    """Clean and optionally uppercase types for strict schema validation."""
    if not isinstance(schema, dict):
        return schema
    allowed_keys = {"type", "format", "description", "enum", "properties", "required", "items"}
    cleaned = {}
    for k, v in schema.items():
        if k in allowed_keys:
            if k == "type" and isinstance(v, str):
                cleaned[k] = v.upper() if for_gemini else v
            elif k == "properties":
                cleaned[k] = {prop_k: clean_schema(prop_v, for_gemini) for prop_k, prop_v in v.items()}
            elif k == "items":
                cleaned[k] = clean_schema(v, for_gemini)
            else:
                cleaned[k] = v
    return cleaned


def convert_tool_schema_to_openai(tool_dict: dict) -> dict:
    """Convert Friday's ToolSchema to OpenAI format (used by llama-cpp)."""
    input_schema = tool_dict.get("input_schema", {})
    if input_schema.get("type") in ("object", "OBJECT"):
        params = clean_schema(input_schema, for_gemini=False)
    else:
        params = clean_schema({
            "type": "object",
            "properties": input_schema
        }, for_gemini=False)
        
    return {
        "type": "function",
        "function": {
            "name": tool_dict["name"],
            "description": tool_dict["description"],
            "parameters": params
        }
    }


def convert_tool_schema_to_gemini(tool_dict: dict) -> dict:
    """Convert Friday's ToolSchema to Gemini format."""
    input_schema = tool_dict.get("input_schema", {})
    if input_schema.get("type") in ("object", "OBJECT"):
        params = clean_schema(input_schema, for_gemini=True)
    else:
        params = clean_schema({
            "type": "OBJECT",
            "properties": input_schema
        }, for_gemini=True)

    return {
        "name": tool_dict["name"],
        "description": tool_dict["description"],
        "parameters": params
    }


async def execute_tool_call(name: str, params: dict) -> str:
    print(f"\n[Tool Execution] {name}({params})")
    try:
        result = await router.dispatch(name, params)
        return str(result)
    except Exception as e:
        return f"Error executing tool: {e}"


async def run_llama_cpp(model_path: str):
    try:
        from llama_cpp import Llama
    except ImportError:
        print("Please install llama-cpp-python.")
        return

    print(f"Loading GGUF model from {model_path}...")
    llm = Llama(model_path=model_path, n_ctx=4096, chat_format="chatml-function-calling")
    
    tools = [convert_tool_schema_to_openai(t) for t in router.list_tools()]
    messages = [{"role": "system", "content": "You are a helpful assistant with access to tools."}]

    while True:
        try:
            user_input = input("\nUser: ")
        except (KeyboardInterrupt, EOFError):
            break
        if not user_input.strip():
            continue
            
        messages.append({"role": "user", "content": user_input})
        
        while True:
            response = llm.create_chat_completion(
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            choice = response["choices"][0]
            message = choice["message"]
            messages.append(message)
            
            if "tool_calls" in message and message["tool_calls"]:
                for tool_call in message["tool_calls"]:
                    func = tool_call["function"]
                    name = func["name"]
                    args = json.loads(func["arguments"])
                    
                    result = await execute_tool_call(name, args)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": name,
                        "content": result
                    })
            else:
                print(f"\nAssistant: {message.get('content', '')}")
                break


async def run_gemini():
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("Please install google-genai. (pip install google-genai)")
        return

    if not os.environ.get("GEMINI_API_KEY"):
        print("Please set GEMINI_API_KEY environment variable.")
        return

    client = genai.Client()
    
    # Create the function declarations array
    func_declarations = [convert_tool_schema_to_gemini(t) for t in router.list_tools()]
    gemini_tools = [{"function_declarations": func_declarations}]
    
    system_instruction = (
        "You are Friday, a helpful desktop AI assistant running on the user's Linux machine. "
        "You have access to the following categories of tools — use them proactively:\n\n"

        "TERMINAL (create_terminal_session, write_to_terminal, read_from_terminal, kill_terminal_session):\n"
        "- You CAN run shell commands on the user's system. When the user asks you to do something "
        "that requires a command (e.g. 'update my system', 'install a package', 'list files', "
        "'run a script'), create a terminal session and execute the appropriate command. "
        "Always read the terminal output after writing a command to check results.\n\n"

        "WEB FETCH (web_fetch):\n"
        "- To read any public URL, ALWAYS use the `web_fetch` tool first. "
        "It works everywhere without any extension.\n\n"

        "BROWSER TOOLS (browser_*):\n"
        "- These tools ONLY work when the Friday Chrome extension is connected. "
        "If a browser tool returns 'extension is not connected', immediately fall back to `web_fetch`. "
        "Do NOT keep retrying browser tools.\n\n"

        "TASKS (create_task, get_tasks, update_task, delete_task):\n"
        "- These manage the user's to-do list, NOT system operations. "
        "Do not confuse 'update my system' (use terminal) with 'update a task' (update_task tool).\n\n"

        "MEMORY (log_event, query_memory) and CALENDAR (get_calendar_events):\n"
        "- Use for storing/retrieving context and checking the user's schedule.\n\n"

        "GENERAL RULES:\n"
        "- Always pick the most capable tool for the job. Never refuse a request if you have a tool that can help.\n"
        "- When a tool returns an error, report it clearly and try an alternative approach.\n"
        "- Ask for confirmation before running destructive commands (rm, format, etc.)."
    )

    chat = client.chats.create(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=gemini_tools,
            temperature=0.7,
        )
    )

    print("Gemini Interactive Session started.")
    
    while True:
        try:
            user_input = input("\nUser: ")
        except (KeyboardInterrupt, EOFError):
            break
        if not user_input.strip():
            continue

        try:
            response = chat.send_message(user_input)
            
            while response.function_calls:
                # Gemini called a tool
                func_calls = response.function_calls
                responses = []
                for func_call in func_calls:
                    name = func_call.name
                    args = {k: v for k, v in func_call.args.items()}
                    
                    result = await execute_tool_call(name, args)
                    
                    responses.append(types.Part.from_function_response(
                        name=name,
                        response={"result": result}
                    ))
                
                response = chat.send_message(responses)
                
            print(f"\nAssistant: {response.text}")
        except Exception as e:
            print(f"Error communicating with Gemini: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run Friday Model Flow")
    parser.add_argument("--model", type=str, choices=["gemini", "llama"], default="gemini",
                        help="Choose which model backend to use")
    parser.add_argument("--model-path", type=str, default="",
                        help="Path to the GGUF model file (if using llama)")
    
    args = parser.parse_args()
    
    if args.model == "llama" and not args.model_path:
        print("Error: --model-path is required when using llama backend")
        sys.exit(1)
        
    if args.model == "gemini":
        asyncio.run(run_gemini())
    else:
        asyncio.run(run_llama_cpp(args.model_path))


if __name__ == "__main__":
    main()
