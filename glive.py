import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

import sys
import asyncio
import base64
import io
import json
import traceback

import cv2
import pyaudio
import PIL.Image

import argparse

from google import genai
from google.genai import types
from google.genai.types import Type

# ── Backend MCP tool wiring ───────────────────────────────────────────────────
# Add workspace root so backend package is importable when running glive.py directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.engine.router import ToolRouter
from backend.model_flow import convert_tool_schema_to_gemini

# Initialise the tool router (same singleton used by the FastAPI backend)
_tool_router = ToolRouter()

def _build_gemini_tools() -> list[types.Tool]:
    """Convert all registered MCP tools to Gemini FunctionDeclaration objects."""
    func_declarations = []
    for tool_dict in _tool_router.list_tools():
        schema = convert_tool_schema_to_gemini(tool_dict)
        func_declarations.append(
            types.FunctionDeclaration(
                name=schema["name"],
                description=schema["description"],
                parameters=schema.get("parameters"),
            )
        )
    return [types.Tool(function_declarations=func_declarations)]


async def _dispatch_tool_call(name: str, args: dict) -> str:
    """Execute a tool call via the MCP router and return a JSON string result."""
    print(f"\n[MCP Tool Call] {name}({args})")
    try:
        result = await _tool_router.dispatch(name, args)
        return json.dumps(result, default=str)
    except Exception as exc:
        error_msg = f"Tool error: {exc}"
        print(f"  ↳ {error_msg}")
        return json.dumps({"error": error_msg})

# ─────────────────────────────────────────────────────────────────────────────

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 44100
RECEIVE_SAMPLE_RATE = 44100
CHUNK_SIZE = 1024

MODEL = "models/gemini-3.1-flash-live-preview"

DEFAULT_MODE = "camera"

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
)


def _build_config() -> types.LiveConnectConfig:
    """Build LiveConnectConfig with MCP tools injected."""
    return types.LiveConnectConfig(
        response_modalities=[
            "AUDIO",
        ],
        media_resolution="MEDIA_RESOLUTION_LOW",
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
            )
        ),
        context_window_compression=types.ContextWindowCompressionConfig(
            trigger_tokens=104857,
            sliding_window=types.SlidingWindow(target_tokens=52428),
        ),
        # Inject all registered MCP tools
        tools=_build_gemini_tools(),
    )


CONFIG = _build_config()

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

        self.audio_stream = None

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            if self.session is not None:
                await self.session.send_client_content(
                    turns=[types.Content(role="user", parts=[types.Part(text=text or ".")])],
                    turn_complete=True,
                )

    def _get_frame(self, cap):
        # Read the frame
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(
            cv2.VideoCapture, 0
        )  # 0 represents the default camera

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            if self.out_queue is not None:
                await self.out_queue.put(frame)

        # Release the VideoCapture object
        cap.release()

    def _get_screen(self):
        try:
            import mss  # pytype: disable=import-error # pylint: disable=g-import-not-at-top
        except ImportError as e:
            raise ImportError("Please install mss package using 'pip install mss'") from e
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):

        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            if self.out_queue is not None:
                await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            if self.out_queue is not None:
                msg = await self.out_queue.get()
                if self.session is not None:
                    mime = msg.get("mime_type", "")
                    raw = msg["data"]
                    if mime == "audio/pcm":
                        # Raw audio bytes from mic
                        await self.session.send_realtime_input(
                            audio=types.Blob(data=raw, mime_type=mime)
                        )
                    elif mime.startswith("image/"):
                        # Base64-encoded image string from camera/screen
                        import base64 as _b64
                        image_bytes = _b64.b64decode(raw) if isinstance(raw, str) else raw
                        await self.session.send_realtime_input(
                            video=types.Blob(data=image_bytes, mime_type=mime)
                        )
                    else:
                        # Fallback: send as client content text
                        await self.session.send_client_content(
                            turns=[types.Content(role="user", parts=[types.Part(text=str(msg))])],
                            turn_complete=False,
                        )

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            if self.out_queue is not None:
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        """Background task: reads from the websocket, handles audio, text,
        and MCP tool calls dispatched by the model."""
        while True:
            if self.session is None:
                await asyncio.sleep(0.1)
                continue

            turn = self.session.receive()
            async for response in turn:
                # ── Audio chunk ───────────────────────────────────────────
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)

                # ── Text (transcript / model text) ────────────────────────
                if text := response.text:
                    print(text, end="", flush=True)

                # ── MCP Tool Call ─────────────────────────────────────────
                if response.tool_call:
                    for fc in response.tool_call.function_calls:
                        # fc.args is a MapComposite (proto map); convert to dict
                        args = dict(fc.args) if fc.args else {}
                        result_str = await _dispatch_tool_call(fc.name, args)
                        print(f"  ↳ result: {result_str[:120]}")

                        # Send the tool response back to the model
                        await self.session.send_tool_response(
                            function_responses=[
                                types.FunctionResponse(
                                    id=fc.id,
                                    name=fc.name,
                                    response={"result": result_str},
                                )
                            ]
                        )

            # If the model finishes its turn (or was interrupted), flush audio.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            if self.audio_in_queue is not None:
                bytestream = await self.audio_in_queue.get()
                await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            if self.audio_stream is not None:
                self.audio_stream.close()
                traceback.print_exception(EG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())
