"""
Benchmark greedy decoding speed of the CTC vs RNN-T checkpoint on a held-out manifest.

Reports RTFx (real-time factor) for both. RNN-T should be slower; the gap shrinks
in exercise 6 with TDT.

Usage:
    python benchmark.py \
        --ctc_checkpoint  ../04-train-fastconformer-ctc/lightning_logs/.../last.ckpt \
        --rnnt_checkpoint lightning_logs/.../last.ckpt \
        --manifest ../04-train-fastconformer-ctc/data/test-clean/test_clean.json \
        --batch_size 16
"""
import argparse
import json
import time

import torch

from nemo.collections.asr.models import EncDecCTCModelBPE, EncDecRNNTBPEModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctc_checkpoint", required=True)
    p.add_argument("--rnnt_checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_files", type=int, default=200,
                   help="cap files for a quick benchmark; set high for accuracy")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = [json.loads(l) for l in open(args.manifest)][:args.max_files]
    paths = [r["audio_filepath"] for r in rows]
    total_audio_sec = sum(r["duration"] for r in rows)

    def bench(model, name):
        model.eval()
        # warmup
        _ = model.transcribe(paths[:args.batch_size], batch_size=args.batch_size)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.transcribe(paths, batch_size=args.batch_size)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        rtfx = total_audio_sec / elapsed
        print(f"  {name:6s}: {total_audio_sec:7.1f}s audio / {elapsed:6.2f}s wall = RTFx {rtfx:7.1f}")

    print(f"Benchmarking on {len(rows)} utterances ({total_audio_sec/60:.1f} min) at batch {args.batch_size}")

    print("\nCTC checkpoint:")
    ctc = EncDecCTCModelBPE.load_from_checkpoint(args.ctc_checkpoint, map_location=device)
    bench(ctc, "CTC")
    del ctc; torch.cuda.empty_cache() if device == "cuda" else None

    print("\nRNN-T checkpoint:")
    rnnt = EncDecRNNTBPEModel.load_from_checkpoint(args.rnnt_checkpoint, map_location=device)
    bench(rnnt, "RNN-T")


if __name__ == "__main__":
    main()
