"""
Pull TDT word-level timestamps out of a checkpoint.

Usage:
    python extract_timestamps.py \
        --checkpoint lightning_logs/.../last.ckpt \
        --audio path/to/clip.wav
"""
import argparse

import torch
from nemo.collections.asr.models import EncDecRNNTBPEModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--audio", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.checkpoint}")
    model = EncDecRNNTBPEModel.restore_from(args.checkpoint, map_location=device)
    model.eval()

    # NeMo's TDT decoders expose timestamps when you request hypotheses.
    hyps = model.transcribe([args.audio], return_hypotheses=True, timestamps=True)
    if isinstance(hyps[0], list):
        hyp = hyps[0][0]
    else:
        hyp = hyps[0]

    print(f"\nTranscript: {hyp.text}\n")
    # word_timestamps if available
    word_ts = getattr(hyp, "timestamps", None)
    if word_ts:
        # NeMo returns a dict {'word': [{'word': ..., 'start_offset': ..., ...}, ...], ...}
        words = word_ts.get("word") if isinstance(word_ts, dict) else word_ts
        if words:
            print(f"{'word':<15s} {'start':>8s} {'end':>8s}")
            for w in words:
                if isinstance(w, dict):
                    start = w.get("start", w.get("start_offset", "?"))
                    end = w.get("end", w.get("end_offset", "?"))
                    print(f"{w.get('word',''):<15s} {str(start):>8s} {str(end):>8s}")
        else:
            print("No word-level timestamps returned. Check your NeMo version supports them in the TDT decoder.")
    else:
        print("Hypothesis has no .timestamps attribute. NeMo version mismatch?")


if __name__ == "__main__":
    main()
