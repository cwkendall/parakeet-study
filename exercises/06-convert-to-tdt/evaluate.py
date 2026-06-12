"""TDT evaluator. Same as exercise 5's but loads as RNNT model with TDT decoding."""
import argparse, json, torch
from nemo.collections.asr.models import EncDecRNNTBPEModel


def wer(reference, hypothesis):
    ref = reference.split(); hyp = hypothesis.split()
    m, n = len(ref), len(hyp)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n], m


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EncDecRNNTBPEModel.restore_from(args.checkpoint, map_location=device)
    model.eval()

    rows = [json.loads(l) for l in open(args.manifest)]
    refs = [r["text"].strip().lower() for r in rows]
    paths = [r["audio_filepath"] for r in rows]
    hyps = model.transcribe(paths, batch_size=args.batch_size)
    if isinstance(hyps[0], list):
        hyps = [h[0].text if hasattr(h[0], "text") else h[0] for h in hyps]

    errs = words = 0
    for r, h in zip(refs, hyps):
        e, n = wer(r, h)
        errs += e; words += n
    print(f"\nTDT WER over {len(rows)} utterances: {errs/words*100:.2f}% ({errs}/{words})")
    for r, h in list(zip(refs, hyps))[:5]:
        print(f"  REF: {r}\n  HYP: {h}\n")
