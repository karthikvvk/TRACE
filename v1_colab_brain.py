# ──────────────────────────────────────────────────────────────────────────────
# Cell 4 : SmolLM3 Tool Calling Brain
# ──────────────────────────────────────────────────────────────────────────false────
# pip install -q sentence-transformers

import asyncio
import gc
import json
import logging
import re

import nest_asyncio
import numpy as np
import os

# Must be set before torch is imported so the CUDA allocator picks it up.
# Expandable segments reduce fragmentation OOM when allocation sizes vary
# across iterations (e.g. web_fetch result vs. short browser tool result).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import websockets
from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("colab_brain_local")

MAX_TOOL_ITERATIONS = 6

# Heavy tools (web_fetch, web_search, browser_get_dom) can return 50K-200K
# chars. Keeping them raw bloats the KV cache on the next generate() call.
# Browser DATA tools (downloads, bookmarks) can reach 6-10K — keep those.
MAX_TOOL_RESULT_CHARS = 8000

# Action tools perform a side-effect and return minimal confirmation.
# After one successful call the model tends to call them again to "verify".
# We break out immediately after the first call and synthesize a confirmation
# instead of running the dedup guard which gives a confusing error message.
ACTION_TOOLS = {
    # Tab / navigation
    "browser_open_tab",
    "browser_close_tab",
    "browser_navigate",
    # DOM interaction
    "browser_click",
    "browser_fill_input",
    "browser_execute_script",
    # Misc browser actions
    "browser_open_settings",
    "browser_print_page",
    # Download actions
    "browser_cancel_download",
    "browser_erase_download",
    "browser_open_download",
    "browser_show_download",
    # Bookmark write actions
    "browser_create_bookmark",
    "browser_delete_bookmark",
    "browser_delete_bookmark_folder",
    "browser_move_bookmark",
    # Task / memory writes
    "create_task",
    "update_task",
    "delete_task",
    "log_event",
    # Terminal writes
    "create_terminal",
    "write_terminal",
    "kill_terminal",
}


# ──────────────────────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"

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


# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are Friday, a helpful AI assistant.

You have access to tools.

Read every tool description carefully.

Use whichever tool best satisfies the user's request.

Prefer using a tool instead of guessing whenever a tool can provide the answer.

Think step by step.

When using a tool you MUST respond ONLY with:

<tool_call>
{
  "name":"tool_name",
  "arguments":{
    ...
  }
}
</tool_call>

Do not include any other text.

If no tool is needed, answer normally.
""".strip()


# ──────────────────────────────────────────────────────────────────────────────
# TOOL CONVERSION
# ──────────────────────────────────────────────────────────────────────────────

def build_tools(tool_schemas):

    tools = []

    for tool in tool_schemas:

        tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            }
        })

    return tools


# ──────────────────────────────────────────────────────────────────────────────
# TOOL EMBEDDER  (CPU-only, no GPU memory used)
# ──────────────────────────────────────────────────────────────────────────────

class ToolEmbedder:
    """
    Embeds all tool descriptions once at startup using a tiny CPU model
    (all-MiniLM-L6-v2, 22 MB).  Per query, returns the top-K most relevant
    tools by cosine similarity so the LLM only sees a small, focused list.
    """

    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, tool_schemas, k: int = 5):

        self.k = k
        self.tools = build_tools(tool_schemas)

        print(f"Loading embedding model ({self.EMBED_MODEL}) on CPU...")
        self._embedder = SentenceTransformer(self.EMBED_MODEL, device="cpu")

        # One text per tool: "name: description"
        texts = [
            f"{t['function']['name']}: {t['function']['description']}"
            for t in self.tools
        ]

        # Shape: (num_tools, embedding_dim) — stays on CPU RAM, not GPU
        self._tool_embeddings = self._embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        print(f"Embedded {len(self.tools)} tools ✅")

    def top_k(self, query: str) -> list:
        """Return the top-k tools most semantically similar to the query."""

        q_emb = self._embedder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Cosine similarity (embeddings are L2-normalised so dot == cosine)
        scores = (self._tool_embeddings @ q_emb.T).squeeze()
        indices = np.argsort(scores)[::-1][: self.k]

        selected = [self.tools[i] for i in indices]

        logger.info(
            "[ToolEmbedder] Selected tools for query: %s",
            [t["function"]["name"] for t in selected],
        )

        return selected


# ──────────────────────────────────────────────────────────────────────────────
# GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def generate(messages, tools):

    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    # .clone() breaks the view so the full `output` tensor (KV cache + all
    # token buffers) can be freed immediately rather than lingering until the
    # next call — which is what causes GPU to fill up on the 2nd query.
    generated = output[0][inputs.input_ids.shape[1]:].clone()

    del output, inputs
    torch.cuda.empty_cache()
    gc.collect()

    return tokenizer.decode(
        generated,
        skip_special_tokens=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def strip_thinking(text):

    return re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL,
    ).strip()


def extract_balanced_json(text):

    start = text.find("{")

    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):

        c = text[i]

        if escape:
            escape = False
            continue

        if c == "\\":
            escape = True
            continue

        if c == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if c == "{":
            depth += 1

        elif c == "}":
            depth -= 1

            if depth == 0:
                return text[start:i + 1]

    return None


# ──────────────────────────────────────────────────────────────────────────────
# TOOL PARSER
# ──────────────────────────────────────────────────────────────────────────────

TOOL_PATTERNS = [
    r"<tool_call>(.*?)</tool_call>",
    r"<tool>(.*?)</tool>",
    r"```json\s*(.*?)\s*```",
]


def _coerce_args(args) -> dict:
    """
    Qwen's chat template occasionally serialises `arguments` as a JSON
    string (e.g. '{"query": "foo"}') rather than a dict.  If the tool
    router receives a string it crashes with:
        'str' object has no attribute 'get'
    This helper always returns a plain dict.
    """
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def parse_tool_call(text):

    for index, pattern in enumerate(TOOL_PATTERNS):

        match = re.search(
            pattern,
            text,
            flags=re.DOTALL,
        )

        if not match:
            continue

        if index > 0:
            logger.warning(
                "Tool call matched via fallback pattern #%d",
                index,
            )

        payload = match.group(1).strip()

        try:

            data = json.loads(payload)

            name = (
                data.get("name")
                or data.get("tool")
                or data.get("function")
            )

            args = _coerce_args(
                data.get("arguments")
                or data.get("args")
                or {}
            )

            if name:
                return name, args

        except Exception:
            pass

    payload = extract_balanced_json(text)

    if payload:

        logger.warning(
            "Tool call matched via balanced JSON fallback"
        )

        try:

            data = json.loads(payload)

            name = (
                data.get("name")
                or data.get("tool")
                or data.get("function")
            )

            args = _coerce_args(
                data.get("arguments")
                or data.get("args")
                or {}
            )

            if name:
                return name, args

        except Exception:
            pass

    return None


# ──────────────────────────────────────────────────────────────────────────────
# AGENT LOOP
# ──────────────────────────────────────────────────────────────────────────────

async def run_turn_local(
    ws,
    turn_id,
    user_message,
    history,
    tool_embedder,
):
    # Retrieve only the top-K relevant tools for this specific query.
    # Passing all 42 tools to a 3B model causes it to pick wrong tools
    # and use fallback output formats.
    tools = tool_embedder.top_k(user_message)

    messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT,
    }]

    for item in history:

        messages.append({
            "role": item.get("role", "user"),
            "content": item.get("content", ""),
        })

    messages.append({
        "role": "user",
        "content": user_message,
    })

    # Track last call for dedup
    last_call_signature = None

    try:

        for iteration in range(MAX_TOOL_ITERATIONS):

            # On the last iteration, inject a nudge so the model synthesizes
            # rather than calling another tool.
            if iteration == MAX_TOOL_ITERATIONS - 1:
                messages.append({
                    "role": "user",
                    "content": (
                        "You now have all the information you need. "
                        "Please provide a clear, direct answer to the user's question "
                        "based on the tool results above. Do NOT call any more tools."
                    ),
                })

            output = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: generate(messages, tools)
            )

            logger.info(
                "[Turn %s] Iteration %d",
                turn_id[:8],
                iteration + 1,
            )

            logger.info(
                "[Turn %s] Output: %s",
                turn_id[:8],
                output[:400],
            )

            tool_call = parse_tool_call(output)

            if tool_call is None:

                cleaned = strip_thinking(output)

                await ws.send(json.dumps({
                    "type": "text_chunk",
                    "id": turn_id,
                    "content": cleaned,
                }))

                break

            tool_name, tool_args = tool_call

            # ── Deduplication guard ────────────────────────────────────────────
            call_signature = json.dumps(
                {"name": tool_name, "args": tool_args},
                sort_keys=True,
            )

            if call_signature == last_call_signature:
                logger.warning(
                    "[Turn %s] Duplicate tool call detected (%s), forcing synthesis",
                    turn_id[:8],
                    tool_name,
                )
                # Force the model to answer with what it already has
                messages.append({
                    "role": "user",
                    "content": (
                        "You already called that tool. "
                        "Use the results you already have to answer the user."
                    ),
                })
                output = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: generate(messages, tools)
                )
                cleaned = strip_thinking(output)
                await ws.send(json.dumps({
                    "type": "text_chunk",
                    "id": turn_id,
                    "content": cleaned,
                }))
                break

            last_call_signature = call_signature
            # ──────────────────────────────────────────────────────────────────

            logger.info(
                "[Turn %s] Tool %s %s",
                turn_id[:8],
                tool_name,
                tool_args,
            )

            await ws.send(json.dumps({
                "type": "tool_call",
                "id": turn_id,
                "name": tool_name,
                "args": tool_args,
            }))

            tool_result_msg = json.loads(
                await ws.recv()
            )

            tool_result = tool_result_msg.get("result", "")

            logger.info(
                "[Turn %s] Tool returned %s",
                turn_id[:8],
                type(tool_result).__name__,
            )

            # ── Proper Qwen tool-call history format ───────────────────────────
            # Set tool_call_id and record the assistant tool-call turn before
            # branching. Both the action-tool early-break path and the
            # data-tool continuation path need this in messages[].
            tool_call_id = f"call_{iteration}"

            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args),
                    },
                }],
            })

            # ── Action tool: synthesize immediately after first call ──────────────
            # Action tools (open_tab, navigate, click, etc.) perform a
            # side-effect and return minimal confirmation. The small model
            # tends to call them again to "verify". Instead we break here
            # with an action-done synthesis nudge so the answer is clean.
            if tool_name in ACTION_TOOLS:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": json.dumps(
                        tool_result if isinstance(tool_result, (dict, list))
                        else {"result": tool_result},
                        ensure_ascii=False,
                    ),
                })
                messages.append({
                    "role": "user",
                    "content": (
                        "The action completed successfully. "
                        "Confirm to the user in one short sentence that it is done."
                    ),
                })
                action_output = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: generate(messages, tools)
                )
                await ws.send(json.dumps({
                    "type": "text_chunk",
                    "id": turn_id,
                    "content": strip_thinking(action_output),
                }))
                break

            # ── Truncate heavy tool results before they enter the context ───────
            # web_fetch / web_search can return 50K+ chars. Keeping them full
            # makes the next generate() call allocate a huge KV cache → OOM.
            # For JSON list results, trim at item boundaries so the model
            # receives valid JSON, not a half-cut array.
            raw_result = (
                tool_result if isinstance(tool_result, (dict, list))
                else {"result": tool_result}
            )
            tool_result_str = json.dumps(raw_result, ensure_ascii=False)

            if len(tool_result_str) > MAX_TOOL_RESULT_CHARS:
                if isinstance(raw_result, list) and len(raw_result) > 1:
                    # Trim list at item boundary so JSON stays valid
                    trimmed = list(raw_result)
                    while len(json.dumps(trimmed, ensure_ascii=False)) > MAX_TOOL_RESULT_CHARS and len(trimmed) > 1:
                        trimmed.pop()
                    omitted = len(raw_result) - len(trimmed)
                    trimmed.append({"_note": f"... and {omitted} more items omitted"})
                    tool_result_str = json.dumps(trimmed, ensure_ascii=False)
                    logger.warning(
                        "[Turn %s] Tool result from '%s' trimmed to %d/%d items",
                        turn_id[:8], tool_name, len(trimmed) - 1, len(raw_result),
                    )
                else:
                    tool_result_str = tool_result_str[:MAX_TOOL_RESULT_CHARS] + '… [truncated]"}'
                    logger.warning(
                        "[Turn %s] Tool result from '%s' truncated (%d → %d chars)",
                        turn_id[:8], tool_name,
                        len(json.dumps(raw_result, ensure_ascii=False)),
                        MAX_TOOL_RESULT_CHARS,
                    )

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": tool_result_str,
            })
            # ──────────────────────────────────────────────────────────────────

        else:

            logger.warning(
                "[Turn %s] Maximum tool iterations reached",
                turn_id[:8],
            )

            await ws.send(json.dumps({
                "type": "error",
                "id": turn_id,
                "content": "Maximum tool iterations reached.",
            }))

    except Exception as exc:

        logger.exception(exc)

        await ws.send(json.dumps({
            "type": "error",
            "id": turn_id,
            "content": str(exc),
        }))

    finally:

        await ws.send(json.dumps({
            "type": "done",
            "id": turn_id,
        }))


# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

async def brain_loop_local():

    url = f"{LOCAL_WS_URL}?secret={COLAB_SECRET}"

    print(f"Connecting to {LOCAL_WS_URL}")

    async with websockets.connect(
        url,
        ping_interval=30,
        ping_timeout=10,
    ) as ws:

        print("Connected")

        schema_message = json.loads(
            await ws.recv()
        )

        tool_schemas = schema_message["tools"]

        print(f"Received {len(tool_schemas)} tools")

        # Build the embedding index (CPU-only, ~0.5 s)
        tool_embedder = ToolEmbedder(tool_schemas, k=5)

        async for raw in ws:

            message = json.loads(raw)

            if message.get("type") != "user_message":
                continue

            await run_turn_local(
                ws=ws,
                turn_id=message["id"],
                user_message=message["message"],
                history=message.get("history", []),
                tool_embedder=tool_embedder,
            )


# ──────────────────────────────────────────────────────────────────────────────
# START
# ──────────────────────────────────────────────────────────────────────────────

print(f"Starting {MODEL_ID}")

try:
    await brain_loop_local()

except KeyboardInterrupt:
    print("Stopped")