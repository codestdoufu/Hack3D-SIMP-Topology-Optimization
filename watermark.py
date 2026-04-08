"""
watermark.py — Digital watermarking for SIMP topology optimization density fields.

Research module for Hack3D / NYU VIP Digital Manufacturing Cybersecurity.

Implements spread-spectrum watermarking in the density field:
  - Embed: modulate a pseudo-random carrier with a binary message
  - Detect: correlate recovered signal with known carrier keys
  - Tamper: simulate adversarial attacks (noise, scaling, zeroing)
  - Verify: compute BER and confidence score
"""

import numpy as np
import hashlib


# ── CORE WATERMARK CLASS ──────────────────────────────────────────────────────

class DensityWatermark:
    """
    Spread-spectrum watermarking for FEM density fields.

    The watermark is embedded as a low-amplitude pseudo-random perturbation
    of the density values, modulated by a secret key.  Detection is performed
    by correlating the recovered perturbation with the known carrier sequence.

    Strength parameter alpha controls the robustness / imperceptibility tradeoff.
    """

    def __init__(self, secret_key: str = "hack3d-nyu-vip-2025", alpha: float = 0.03):
        self.secret_key = secret_key
        self.alpha = alpha          # Embedding strength (0.01 = subtle, 0.05 = robust)
        self._rng_seed = self._key_to_seed(secret_key)

    # ── key utilities ──────────────────────────────────────────────────────────

    @staticmethod
    def _key_to_seed(key: str) -> int:
        """Deterministically convert a string key to an integer RNG seed."""
        return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)

    def _make_carrier(self, n: int) -> np.ndarray:
        """Generate a bipolar (±1) pseudo-random carrier sequence from the secret key."""
        rng = np.random.default_rng(self._rng_seed)
        return rng.choice([-1.0, 1.0], size=n)

    # ── message encoding ───────────────────────────────────────────────────────

    @staticmethod
    def text_to_bits(text: str, n_bits: int = 64) -> np.ndarray:
        """Encode a string as a fixed-length bit array."""
        raw = text.encode("utf-8")
        bits = []
        for byte in raw:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        bits = bits[:n_bits]
        while len(bits) < n_bits:
            bits.append(0)
        return np.array(bits, dtype=float)

    @staticmethod
    def bits_to_text(bits: np.ndarray) -> str:
        """Decode a bit array back to a string (best-effort)."""
        chars = []
        for i in range(0, len(bits) - 7, 8):
            byte = 0
            for b in bits[i:i+8]:
                byte = (byte << 1) | int(round(b))
            if byte == 0:
                break
            try:
                chars.append(chr(byte))
            except Exception:
                chars.append("?")
        return "".join(chars)

    # ── embed ──────────────────────────────────────────────────────────────────

    def embed(self, density: np.ndarray, message: str = "NYU-HACK3D") -> dict:
        """
        Embed watermark into density field using spread-spectrum modulation.

        Each message bit is spread over n_elem / n_bits elements using
        the carrier sequence, then added at amplitude alpha.

        Returns dict with:
          - watermarked_density: modified density array
          - original_density: copy of original
          - message: embedded message string
          - bits: binary representation
          - perturbation: the actual delta added
          - snr_db: signal-to-noise ratio in dB
        """
        n = len(density)
        bits = self.text_to_bits(message, n_bits=min(64, n // 4))
        carrier = self._make_carrier(n)

        # Spread each bit across a segment of the carrier
        n_bits = len(bits)
        segment = n // n_bits
        watermark_signal = np.zeros(n)
        for i, bit in enumerate(bits):
            symbol = 2 * bit - 1          # map 0→-1, 1→+1
            start = i * segment
            end = min(start + segment, n)
            watermark_signal[start:end] = symbol * carrier[start:end]

        # Scale to alpha and add
        perturbation = self.alpha * watermark_signal
        watermarked = np.clip(density + perturbation, 0.0, 1.0)

        # Compute SNR
        signal_power = np.mean(perturbation ** 2)
        noise_power  = np.mean((watermarked - density) ** 2) + 1e-12
        snr_db = 10 * np.log10(signal_power / noise_power + 1e-12)

        return {
            "watermarked_density": watermarked,
            "original_density":    density.copy(),
            "message":             message,
            "bits":                bits.tolist(),
            "perturbation":        perturbation.tolist(),
            "snr_db":              round(float(snr_db), 2),
            "alpha":               self.alpha,
            "n_bits":              int(n_bits),
        }

    # ── detect ─────────────────────────────────────────────────────────────────

    def detect(self, density: np.ndarray, original: np.ndarray = None,
               n_bits: int = 64) -> dict:
        """
        Detect and decode watermark from density field.

        If original is provided, uses the difference directly (informed detection).
        Otherwise uses blind detection via carrier correlation.

        Returns dict with:
          - detected_message: decoded string
          - detected_bits: recovered bit array
          - confidence: per-bit correlation strength (0–1)
          - ber: bit error rate vs. expected (if original known)
          - is_watermarked: boolean verdict
          - correlation_score: overall detection strength
        """
        n = len(density)
        carrier = self._make_carrier(n)
        n_bits = min(n_bits, n // 4)
        segment = n // n_bits

        if original is not None:
            # Informed: use the actual perturbation
            perturbation = density - original
        else:
            # Blind: high-pass the density to isolate the watermark
            kernel_size = max(3, segment // 4)
            smoothed = np.convolve(density, np.ones(kernel_size) / kernel_size, mode="same")
            perturbation = density - smoothed

        # Correlate each segment with the carrier
        detected_bits = []
        confidences   = []
        for i in range(n_bits):
            start = i * segment
            end   = min(start + segment, n)
            seg_wm = perturbation[start:end]
            seg_c  = carrier[start:end]
            corr = float(np.dot(seg_wm, seg_c)) / (np.linalg.norm(seg_wm) * np.linalg.norm(seg_c) + 1e-12)
            detected_bits.append(1 if corr > 0 else 0)
            confidences.append(abs(corr))

        detected_bits = np.array(detected_bits)
        avg_confidence = float(np.mean(confidences))
        is_watermarked = avg_confidence > 0.05

        detected_message = self.bits_to_text(detected_bits)

        return {
            "detected_message":  detected_message,
            "detected_bits":     detected_bits.tolist(),
            "confidence":        confidences,
            "avg_confidence":    round(avg_confidence, 4),
            "is_watermarked":    bool(is_watermarked),
            "correlation_score": round(avg_confidence * 100, 1),
        }

    # ── tamper simulation ──────────────────────────────────────────────────────

    def simulate_attack(self, density: np.ndarray, attack: str, **kwargs) -> dict:
        """
        Simulate an adversarial attack on the watermarked density field.

        Supported attacks:
          - 'noise'    : additive Gaussian noise (sigma=0.05)
          - 'scale'    : multiply all densities by factor (factor=0.9)
          - 'zero'     : zero out a fraction of elements (fraction=0.2)
          - 'quantize' : quantize to n_levels (n_levels=5)
          - 'smooth'   : moving-average smoothing (window=5)

        Returns dict with attacked density and attack metadata.
        """
        attacked = density.copy()

        if attack == "noise":
            sigma = kwargs.get("sigma", 0.05)
            noise = np.random.default_rng(42).normal(0, sigma, size=len(density))
            attacked = np.clip(attacked + noise, 0.0, 1.0)
            meta = {"attack": "Gaussian Noise", "sigma": sigma}

        elif attack == "scale":
            factor = kwargs.get("factor", 0.9)
            attacked = np.clip(attacked * factor, 0.0, 1.0)
            meta = {"attack": "Density Scaling", "factor": factor}

        elif attack == "zero":
            fraction = kwargs.get("fraction", 0.2)
            rng = np.random.default_rng(99)
            idx = rng.choice(len(density), size=int(fraction * len(density)), replace=False)
            attacked[idx] = 0.0
            meta = {"attack": "Random Zeroing", "fraction": fraction}

        elif attack == "quantize":
            n_levels = kwargs.get("n_levels", 5)
            levels = np.linspace(0, 1, n_levels)
            attacked = levels[np.argmin(np.abs(attacked[:, None] - levels[None, :]), axis=1)]
            meta = {"attack": "Quantization", "n_levels": n_levels}

        elif attack == "smooth":
            window = kwargs.get("window", 5)
            attacked = np.convolve(attacked, np.ones(window) / window, mode="same")
            attacked = np.clip(attacked, 0.0, 1.0)
            meta = {"attack": "Smoothing", "window": window}

        else:
            meta = {"attack": "none"}

        # Compute L2 distortion
        distortion = float(np.sqrt(np.mean((attacked - density) ** 2)))
        meta["distortion_rms"] = round(distortion, 5)

        return {"attacked_density": attacked, "meta": meta}