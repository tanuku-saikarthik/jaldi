"""Open audio turn-detection model as one feature (P(turn complete) on the audio prefix).

Default: the open Smart Turn Hinglish ONNX (raw 16 kHz waveform in, last 8 s, log-mel baked in).
Also handles pipecat smart-turn style ONNX graphs that take Whisper log-mel features, by checking
the graph's input shape. Not exercised in the CPU smoke run.
"""

from __future__ import annotations

import numpy as np

DEFAULT_REPO = "codewithmoin/smart-turn-hinglish"
DEFAULT_FILE = "smart_turn_8s.onnx"
WINDOW = 8 * 16000


class TurnDetector:
    def __init__(self, repo: str = DEFAULT_REPO, filename: str = DEFAULT_FILE, providers=None):
        import onnxruntime as ort  # noqa: PLC0415
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        path = hf_hub_download(repo, filename)
        self.sess = ort.InferenceSession(
            path, providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.rank = len(inp.shape)
        self.fe = None
        if self.rank == 3:  # (batch, n_mels, frames): Whisper log-mel features
            from transformers import WhisperFeatureExtractor  # noqa: PLC0415

            self.fe = WhisperFeatureExtractor(chunk_length=8)

    def __call__(self, wav_16k: np.ndarray) -> float:
        x = wav_16k[-WINDOW:].astype(np.float32)
        buf = np.zeros(WINDOW, np.float32)
        buf[WINDOW - len(x):] = x
        if self.fe is not None:
            feats = self.fe(buf, sampling_rate=16000, return_tensors="np",
                            padding="max_length", max_length=WINDOW, truncation=True)
            inp = feats.input_features.astype(np.float32)
        else:
            inp = buf[None, :]
        out = float(np.asarray(self.sess.run(None, {self.input_name: inp})[0]).reshape(-1)[0])
        if out < 0 or out > 1:
            out = float(1 / (1 + np.exp(-out)))
        return out
