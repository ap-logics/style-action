"""
Does a style direction survive if we stop mean-pooling over time?

The objection: three of the four generator sites are mean-pooled over the
sequence, manner is inherently dynamical (tempo, amplitude envelope, jerk), and
a mean over time is exactly the operation that destroys second-moment structure
while leaving first-moment structure alone. On that reading the paper has shown
only that style is not a static time-invariant offset, which is a much weaker
claim than the one it makes.

T2M-GPT is the one system where this is checkable from saved artefacts: the
extraction saved the generated token sequence per cell, so the full
pre-pooling latent can be rebuilt by dequantising through the VQ codebook and
descriptors other than the global mean can be computed on it.

Descriptors compared, all on the same runs and the same cells:

  mean          the paper's site: mean over time of the dequantised codes.
  meanstd       mean concatenated with per-dimension standard deviation, so
                second-moment structure over time is retained.
  thirds        early, middle and late segment means concatenated, so coarse
                temporal ordering is retained.
  delta         mean of the first difference over time, a pure rate descriptor.
  full_flat     first K frames flattened, no aggregation at all.

If consistency rises materially under any of these, the collapse is a property
of the readout rather than of the representation, and the paper's framing has
to change. If it does not, the pooling objection is answered on this model.

Usage:
  python analysis/unpooled_descriptor.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "metrics"))
from style_vectors import style_vectors, consistency   # noqa: E402


def dequantise(tokens: list[int], codebook: np.ndarray) -> np.ndarray:
    """token indices -> (T', 512) sequence of codebook vectors."""
    idx = np.asarray(tokens, dtype=int)
    idx = idx[(idx >= 0) & (idx < len(codebook))]
    return codebook[idx]


def describe(seq: np.ndarray, kind: str, k: int = 8) -> np.ndarray:
    if seq.shape[0] == 0:
        seq = np.zeros((1, 512), dtype=np.float32)
    if kind == "mean":
        return seq.mean(0)
    if kind == "meanstd":
        return np.concatenate([seq.mean(0), seq.std(0)])
    if kind == "thirds":
        n = seq.shape[0]
        cuts = [seq[: max(1, n // 3)], seq[n // 3: max(1, 2 * n // 3)], seq[2 * n // 3:]]
        return np.concatenate([c.mean(0) if len(c) else seq.mean(0) for c in cuts])
    if kind == "delta":
        d = np.diff(seq, axis=0) if seq.shape[0] > 1 else np.zeros((1, seq.shape[1]))
        return d.mean(0)
    if kind == "full_flat":
        out = np.zeros((k, seq.shape[1]), dtype=seq.dtype)
        m = min(k, seq.shape[0])
        out[:m] = seq[:m]
        return out.reshape(-1)
    raise ValueError(kind)


def main() -> None:
    codebook = np.load(ROOT / "vq_codebook.npy")
    runs = sorted((ROOT / "results/t2mgpt_v2").glob("seed*/0"))
    runs = [r for r in runs if (r / "tokens.json").exists()]
    assert runs, "no t2mgpt_v2 runs with tokens.json"
    print(f"{len(runs)} runs, codebook {codebook.shape}\n")

    kinds = ["mean", "meanstd", "thirds", "delta", "full_flat"]
    out = {}
    print(f"{'descriptor':12}{'dim':>7}{'consistency':>14}{'sd over seeds':>16}")
    print("-" * 49)
    for kind in kinds:
        vals = []
        for r in runs:
            tok = json.loads((r / "tokens.json").read_text())
            neutral = [describe(dequantise(t, codebook), kind) for t in tok["neutral"]]
            Z_S = np.stack(neutral)
            Z_T = np.stack([
                np.stack([describe(dequantise(t, codebook), kind) for t in row])
                for row in tok["styled"]
            ])
            vals.append(float(consistency(style_vectors(Z_S, Z_T))[0].mean()))
        out[kind] = dict(consistency=round(float(np.mean(vals)), 4),
                         sd=round(float(np.std(vals)), 4),
                         dim=int(len(neutral[0])))
        print(f"{kind:12}{len(neutral[0]):>7}{np.mean(vals):>14.4f}{np.std(vals):>16.4f}")

    base = out["mean"]["consistency"]
    print(f"\nPaper's reported T2M-GPT consistency at this site: 0.004")
    print(f"Rebuilt from tokens with the same descriptor:      {base:.4f}")
    print("\nAny descriptor that retains temporal structure and lifts consistency")
    print("materially above the mean row would make the collapse a readout effect.")
    (ROOT / "results/unpooled_descriptor.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
