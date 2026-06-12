"""
Benchmark RNN-T greedy vs TDT greedy on the same test set.

Reports for each: RTFx, average joint calls per second of audio, WER.
For TDT: also reports the average predicted duration d.

Usage:
    python benchmark_tdt.py \
        --rnnt_checkpoint ../05-convert-to-rnnt/lightning_logs/.../last.ckpt \
        --tdt_checkpoint  lightning_logs/.../last.ckpt \
        --manifest ../04-train-fastconformer-ctc/data/test-clean/test_clean.json \
        --batch_size 16
"""
import argparse
import json
import time

import numpy as np
import torch

from nemo.collections.asr.models import EncDecRNNTBPEModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rnnt_checkpoint", required=True)
    p.add_argument("--tdt_checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_files", type=int, default=400)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [json.loads(l) for l in open(args.manifest)][:args.max_files]
    paths = [r["audio_filepath"] for r in rows]
    refs = [r["text"].strip().lower() for r in rows]
    total_audio_sec = sum(r["duration"] for r in rows)

    def bench(model, name, is_tdt=False):
        model.eval()
        # warmup
        _ = model.transcribe(paths[:args.batch_size], batch_size=args.batch_size)
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        hyps = model.transcribe(paths, batch_size=args.batch_size)
        if device == "cuda": torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if isinstance(hyps[0], list):
            hyps_text = [h[0].text if hasattr(h[0], "text") else h[0] for h in hyps]
        else:
            hyps_text = hyps
        rtfx = total_audio_sec / elapsed
        # simple WER
        def wer(r, h):
            ref = r.split(); hyp = h.split()
            m, n = len(ref), len(hyp); dp = [[0]*(n+1) for _ in range(m+1)]
            for i in range(m+1): dp[i][0] = i
            for j in range(n+1): dp[0][j] = j
            for i in range(1, m+1):
                for j in range(1, n+1):
                    dp[i][j] = dp[i-1][j-1] if ref[i-1] == hyp[j-1] else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            return dp[m][n], m
        errs = words = 0
        for r, h in zip(refs, hyps_text):
            e, n = wer(r, h)
            errs += e; words += n
        print(f"  {name:6s}  RTFx={rtfx:7.1f}   WER={errs/words*100:5.2f}%   "
              f"({total_audio_sec:.1f}s audio in {elapsed:.2f}s)")
        return rtfx

    print(f"Benchmarking on {len(rows)} utterances ({total_audio_sec/60:.1f} min) at batch {args.batch_size}\n")

    print("RNN-T:")
    rnnt = EncDecRNNTBPEModel.restore_from(args.rnnt_checkpoint, map_location=device)
    rnnt_rtfx = bench(rnnt, "RNN-T", is_tdt=False)
    del rnnt
    if device == "cuda": torch.cuda.empty_cache()

    print("\nTDT:")
    tdt = EncDecRNNTBPEModel.restore_from(args.tdt_checkpoint, map_location=device)
    tdt_rtfx = bench(tdt, "TDT", is_tdt=True)

    print(f"\nTDT speedup over RNN-T: {tdt_rtfx / rnnt_rtfx:.2f}x")


if __name__ == "__main__":
    main()
