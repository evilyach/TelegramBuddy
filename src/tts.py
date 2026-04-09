import asyncio
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from pydub import AudioSegment


class TTSService:
    """
    Wraps OmniVoice for voice cloning TTS.

    The model is loaded once at process startup (it is heavy).
    `synthesize` runs inference in a thread-pool executor to avoid
    blocking the aiogram event loop during generation.

    Output is OGG/Opus — the format Telegram requires for voice messages
    with waveform visualization. We pre-load the reference audio with
    soundfile to bypass omnivoice's internal torchaudio.load call, which
    breaks on torchaudio >= 2.9 (requires the missing torchcodec package).
    """

    def __init__(
        self, ref_audio: str, ref_text: str, device: str = "cpu", denoise: bool = True
    ):
        self._ref_text = ref_text
        self._generation_config = OmniVoiceGenerationConfig(denoise=denoise)
        self._model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice", device_map=device, dtype=torch.float16
        )

        # Load reference audio with soundfile to avoid torchaudio.load / torchcodec
        audio_np, sr = sf.read(ref_audio, dtype="float32", always_2d=False)
        if audio_np.ndim == 1:
            waveform = torch.from_numpy(audio_np).unsqueeze(0)  # (1, T)
        else:
            waveform = torch.from_numpy(audio_np.T)  # (C, T)
        self._ref_audio: tuple[torch.Tensor, int] = (waveform, sr)

    async def synthesize(self, text: str) -> Path:
        """
        Generate speech for `text`. Returns a path to an OGG/Opus temp file.
        The caller is responsible for deleting it after sending.
        """
        loop = asyncio.get_running_loop()
        audios = await loop.run_in_executor(
            None,
            lambda: self._model.generate(
                text=text,
                ref_audio=self._ref_audio,
                language="Russian",
                ref_text=self._ref_text,
                generation_config=self._generation_config,
            ),
        )
        # audios[0]: (1, T) float32 tensor at self._model.sampling_rate
        audio_np = audios[0].squeeze().numpy()
        sr = self._model.sampling_rate

        # Convert float32 [-1, 1] → int16 for pydub
        audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
        segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sr,
            sample_width=2,  # 16-bit
            channels=1,
        )

        # Export as OGG/Opus — required by Telegram for waveform display
        ogg_path = Path(tempfile.mktemp(suffix=".ogg"))
        segment.export(str(ogg_path), format="ogg", codec="libopus")
        return ogg_path
