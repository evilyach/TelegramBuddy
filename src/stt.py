import base64
import io
import logging

import httpx
from pydub import AudioSegment

logger = logging.getLogger(__name__)

_OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


class STTService:
    """
    Speech-to-text via OpenRouter chat completions with audio-capable models
    (e.g. openai/gpt-4o-audio-preview). Encodes the audio as base64 and sends
    it as a user message content part — avoids the broken /audio/transcriptions
    endpoint on OpenRouter.
    """

    def __init__(self, api_key: str, model: str = "openai/gpt-4o-audio-preview") -> None:
        self._api_key = api_key
        self._model = model

    @staticmethod
    def _to_mp3(audio_bytes: bytes) -> bytes:
        segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="ogg")
        buf = io.BytesIO()
        segment.export(buf, format="mp3")
        return buf.getvalue()

    async def transcribe(self, audio_bytes: bytes) -> str:
        mp3_bytes = self._to_mp3(audio_bytes)
        audio_b64 = base64.b64encode(mp3_bytes).decode()
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_b64, "format": "mp3"},
                        },
                        {
                            "type": "text",
                            "text": "Transcribe this audio exactly as spoken. Return only the transcript, nothing else.",
                        },
                    ],
                }
            ],
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                _OPENROUTER_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if not response.is_success:
                logger.warning("STT request failed: %s — %s", response.status_code, response.text)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
