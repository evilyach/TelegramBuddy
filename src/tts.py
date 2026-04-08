import asyncio
import tempfile
from pathlib import Path

import torchaudio
from omnivoice import OmniVoice


class TTSService:
    """
    Wraps OmniVoice for voice cloning TTS.

    The model is loaded once at process startup (it is heavy).
    `synthesize` runs inference in a thread-pool executor to avoid
    blocking the aiogram event loop during generation.
    """

    def __init__(self, ref_audio: str, ref_text: str, device: str = "cpu"):
        self._ref_audio = ref_audio
        self._ref_text = ref_text
        self._model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device)

    async def synthesize(self, text: str) -> Path:
        """
        Generate speech for `text`. Writes it to a temp WAV file and
        returns the path. The caller is responsible for deleting the file
        after sending it.
        """
        loop = asyncio.get_running_loop()
        audios = await loop.run_in_executor(
            None,
            lambda: self._model.generate(
                text=text,
                ref_audio=self._ref_audio,
                ref_text=self._ref_text,
            ),
        )
        tmp = Path(tempfile.mktemp(suffix=".wav"))
        torchaudio.save(str(tmp), audios[0], self._model.sampling_rate)
        return tmp
