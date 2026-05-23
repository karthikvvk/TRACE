from google.genai import types

def clean_schema(schema: dict) -> dict:
    if not isinstance(schema, dict):
        return schema
    allowed_keys = {"type", "format", "description", "enum", "properties", "required", "items"}
    cleaned = {}
    for k, v in schema.items():
        if k in allowed_keys:
            if k == "type" and isinstance(v, str):
                cleaned[k] = v.upper()
            elif k == "properties":
                cleaned[k] = {prop_k: clean_schema(prop_v) for prop_k, prop_v in v.items()}
            elif k == "items":
                cleaned[k] = clean_schema(v)
            else:
                cleaned[k] = v
    return cleaned

test_schema = {
    "title": {"type": "string", "description": "Short task title"},
    "description": {"type": "string", "description": "Optional detail", "nullable": True},
    "due": {"type": "string", "format": "datetime", "description": "ISO-8601 due datetime", "nullable": True},
    "priority": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"},
    "tags": {"type": "array", "items": {"type": "string"}, "default": []}
}

gemini_params = clean_schema({
    "type": "OBJECT",
    "properties": test_schema
})

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
