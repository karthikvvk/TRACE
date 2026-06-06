# ── Cell 4: Open-Source Brain (Qwen / DeepSeek on Colab GPU) ─────────────────
# Use this INSTEAD of Cell 3 if you want a local model instead of Gemini.
# Requires a GPU runtime in Colab (Runtime → Change runtime type → T4 GPU)

# ── Install (run once) ────────────────────────────────────────────────────────
# !pip install -q transformers accelerate bitsandbytes websockets nest_asyncio

import asyncio, json, logging
import nest_asyncio
nest_asyncio.apply()
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("colab_brain_local")

# ── Config (same as Cell 2) ───────────────────────────────────────────────────
# LOCAL_WS_URL and COLAB_SECRET must already be set from Cell 2

# Pick your model — these all run free on a Colab T4 GPU:
#   "Qwen/Qwen2.5-7B-Instruct"      ← good balance of speed + quality
#   "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  ← strong reasoning
#   "mistralai/Mistral-7B-Instruct-v0.3"
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

MODEL_ID = "HuggingFaceTB/SmolLM3-3B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print(f"Loading {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Model loaded ✅")

# ── Tool-aware generation ─────────────────────────────────────────────────────
# Qwen2.5 supports native function-calling via its chat template.
# We format tools and messages using the tokenizer's apply_chat_template.

def _build_tools_for_qwen(tool_schemas: list[dict]) -> list[dict]:
    """Convert Friday tool schemas → Qwen function-call format."""
    tools = []
    for t in tool_schemas:
        schema = t.get("input_schema", {})
        tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": schema,
            }
        })
    return tools


def _generate(messages, tools):

    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = output[0][inputs.input_ids.shape[1]:]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

def _parse_tool_call(output: str) -> tuple[str, dict] | None:
    """
    Try to extract a tool call from the model output.
    Qwen emits tool calls as JSON inside <tool_call>...</tool_call> tags.
    Returns (name, args) or None if no tool call found.
    """
    import re
    match = re.search(r"<tool_call>(.*?)</tool_call>", output, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
        return data.get("name"), data.get("arguments", {})
    except Exception:
        return None


SYSTEM_PROMPT = (
    "You are Friday, a helpful desktop AI assistant on the user's Linux machine. "
    "Use tools proactively. For system tasks use the terminal tools. "
    "For URLs use web_fetch. Never refuse if you have a capable tool."
)


async def run_turn_local(ws, turn_id: str, message: str, history: list[dict], tools: list[dict]) -> None:
    """Run one agent turn using the local Qwen model."""
    # Build message history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
    messages.append({"role": "user", "content": message})

    try:
        # Agentic loop — keep going until no more tool calls
        while True:
            output = await asyncio.get_event_loop().run_in_executor(
                None, lambda: _generate(messages, tools)
            )
            logger.info("[Turn %s] Model output: %s…", turn_id[:8], output[:120])

            parsed = _parse_tool_call(output)
            if parsed:
                name, args = parsed
                logger.info("[Turn %s] Tool call: %s(%s)", turn_id[:8], name, args)

                # Send tool_call to local machine
                await ws.send(json.dumps({
                    "type": "tool_call", "id": turn_id,
                    "name": name, "args": args,
                }))

                # Wait for result
                result_msg = json.loads(await ws.recv())
                result_str = result_msg.get("result", "")
                logger.info("[Turn %s] Tool result: %s…", turn_id[:8], result_str[:120])

                # Feed result back into context and loop again
                messages.append({"role": "assistant", "content": output})
                messages.append({
                    "role": "tool",
                    "name": name,
                    "content": result_str,
                })
            else:
                # No tool call — this is the final text response
                await ws.send(json.dumps({
                    "type": "text_chunk", "id": turn_id, "content": output
                }))
                break

    except Exception as exc:
        logger.exception("[Turn %s] Error: %s", turn_id[:8], exc)
        await ws.send(json.dumps({"type": "error", "id": turn_id, "content": str(exc)}))
    finally:
        await ws.send(json.dumps({"type": "done", "id": turn_id}))


async def brain_loop_local():
    url = f"{LOCAL_WS_URL}?secret={COLAB_SECRET}"
    print(f"\n🧠 Connecting to Friday at {LOCAL_WS_URL} …")

    async with websockets.connect(url, ping_interval=30, ping_timeout=10) as ws:
        print("✅ Connected! Waiting for tool schemas …")

        schemas_msg = json.loads(await ws.recv())
        tool_schemas = schemas_msg["tools"]
        qwen_tools = _build_tools_for_qwen(tool_schemas)
        print(f"📦 {len(qwen_tools)} tools received. Brain is live 🟢\n")

        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "user_message":
                await run_turn_local(
                    ws, msg["id"], msg["message"], msg.get("history", []), qwen_tools
                )


# ── Run ───────────────────────────────────────────────────────────────────────
print(f"Starting local brain with {MODEL_ID} …")
try:
    await brain_loop_local()
except KeyboardInterrupt:
    print("\n⛔ Brain stopped.")

