from google.genai import types
import json

def clean_schema(schema: dict) -> dict:
    if not isinstance(schema, dict):
        return schema
    allowed_keys = {"type", "format", "description", "enum", "properties", "required", "items"}
    cleaned = {}
    for k, v in schema.items():
        if k in allowed_keys:
            if k == "type" and isinstance(v, str):
                cleaned[k] = v.upper()
            elif isinstance(v, dict):
                cleaned[k] = clean_schema(v)
            elif k == "properties":
                cleaned[k] = {prop_k: clean_schema(prop_v) for prop_k, prop_v in v.items()}
            elif k == "items":
                cleaned[k] = clean_schema(v)
            else:
                cleaned[k] = v
    return cleaned

test_schema = {
    "type": "object",
    "properties": {
        "session_id": {"type": "string"},
        "text": {"type": "string"}
    },
    "required": ["session_id", "text"]
}

gemini_params = clean_schema(test_schema)
print("Gemini Params:", json.dumps(gemini_params, indent=2))

try:
    config = types.GenerateContentConfig(
        tools=[{"function_declarations": [
            {"name": "test", "description": "test desc", "parameters": gemini_params}
        ]}],
        temperature=0.7,
    )
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
