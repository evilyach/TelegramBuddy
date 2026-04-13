import asyncio
import base64
import logging
import tempfile
from pathlib import Path

import numpy as np
import runpod
import soundfile as sf
import torch
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from pydub import AudioSegment

logger = logging.getLogger(__name__)


class TTSService:
    """
    Voice cloning TTS with RunPod as primary backend and local OmniVoice as fallback.

    If `runpod_api_key` and `runpod_endpoint_id` are provided, synthesis is
    offloaded to the RunPod serverless worker. On any error the service falls
    back to local OmniVoice inference transparently.

    The local OmniVoice model is loaded lazily on first use and cached at the
    class level (one instance per device) to avoid redundant loading.
    """

    _model_cache: dict[str, OmniVoice] = {}

    def __init__(
        self,
        ref_audio: str,
        ref_text: str,
        device: str = "cpu",
        denoise: bool = True,
        runpod_api_key: str = "",
        runpod_endpoint_id: str = "",
    ):
        self._ref_audio_path = ref_audio
        self._ref_text = ref_text
        self._device = device
        self._denoise = denoise

        # RunPod — optional
        self._endpoint: runpod.Endpoint | None = None
        if runpod_api_key and runpod_endpoint_id:
            runpod.api_key = runpod_api_key
            self._endpoint = runpod.Endpoint(runpod_endpoint_id)
            logger.info("TTS: RunPod endpoint configured (%s), local inference as fallback", runpod_endpoint_id)
        else:
            logger.info("TTS: no RunPod endpoint configured, using local inference")

        # Local model — loaded lazily
        self._local_model: OmniVoice | None = None
        self._local_ref_audio: tuple[torch.Tensor, int] | None = None
        self._generation_config = OmniVoiceGenerationConfig(denoise=denoise)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_local_model(self) -> None:
        if self._local_model is not None:
            return
        if self._device not in TTSService._model_cache:
            logger.info("TTS: loading local OmniVoice model on device=%s", self._device)
            TTSService._model_cache[self._device] = OmniVoice.from_pretrained(
                "k2-fsa/OmniVoice", device_map=self._device, dtype=torch.float16
            )
        self._local_model = TTSService._model_cache[self._device]

        # Load reference audio with soundfile to avoid torchaudio.load / torchcodec
        audio_np, sr = sf.read(self._ref_audio_path, dtype="float32", always_2d=False)
        if audio_np.ndim == 1:
            waveform = torch.from_numpy(audio_np).unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio_np.T)
        self._local_ref_audio = (waveform, sr)

    def _local_synthesize_sync(self, text: str) -> bytes:
        """Blocking local inference — run in an executor."""
        self._ensure_local_model()
        assert self._local_model is not None
        assert self._local_ref_audio is not None

        audios = self._local_model.generate(
            text=text,
            ref_audio=self._local_ref_audio,
            language="Russian",
            ref_text=self._ref_text,
            generation_config=self._generation_config,
        )
        audio_np = audios[0].squeeze().numpy()
        sr = self._local_model.sampling_rate

        audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
        segment = AudioSegment(audio_int16.tobytes(), frame_rate=sr, sample_width=2, channels=1)

        ogg_path = Path(tempfile.mktemp(suffix=".ogg"))
        segment.export(str(ogg_path), format="ogg", codec="libopus")
        data = ogg_path.read_bytes()
        ogg_path.unlink(missing_ok=True)
        return data

    async def _runpod_synthesize(self, text: str) -> bytes:
        assert self._endpoint is not None
        with open(self._ref_audio_path, "rb") as f:
            ref_audio_b64 = base64.b64encode(f.read()).decode()

        payload = {
            "input": {
                "text": text,
                "ref_audio_base64": ref_audio_b64,
                "ref_text": self._ref_text,
                "denoise": self._denoise,
            }
        }
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: self._endpoint.run_sync(payload, timeout=120)
        )
        if "error" in result:
            raise RuntimeError(f"RunPod TTS error: {result['error']}")
        return base64.b64decode(result["audio_base64"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def synthesize(self, text: str) -> Path:
        """
        Generate speech for `text`. Returns a path to an OGG/Opus temp file.
        The caller is responsible for deleting it after sending.
        """
        ogg_bytes: bytes | None = None

        if self._endpoint is not None:
            try:
                ogg_bytes = await self._runpod_synthesize(text)
                logger.debug("TTS: RunPod synthesis succeeded")
            except Exception as e:
                logger.warning("TTS: RunPod failed (%s), falling back to local inference", e)

        if ogg_bytes is None:
            loop = asyncio.get_running_loop()
            ogg_bytes = await loop.run_in_executor(None, self._local_synthesize_sync, text)
            logger.debug("TTS: local inference succeeded")

        ogg_path = Path(tempfile.mktemp(suffix=".ogg"))
        ogg_path.write_bytes(ogg_bytes)
        return ogg_path
