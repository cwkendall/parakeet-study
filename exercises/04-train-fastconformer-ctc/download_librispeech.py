"""
Download the LibriSpeech subsets required for exercise 4.

Wraps NeMo's get_librispeech_data.py with proxy support (this machine uses
http://127.0.0.1:9000). Skip the proxy block if you don't need it.

Usage:
    python download_librispeech.py --data_root ./data \
        --data_sets train-clean-100,dev-clean,test-clean
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data")
    p.add_argument("--data_sets", default="train-clean-100,dev-clean,test-clean")
    p.add_argument("--use_proxy", action="store_true",
                   help="Use HTTP proxy at http://127.0.0.1:9000 (corp network)")
    args = p.parse_args()

    env = os.environ.copy()
    if args.use_proxy:
        env["http_proxy"] = "http://127.0.0.1:9000"
        env["https_proxy"] = "http://127.0.0.1:9000"

    os.makedirs(args.data_root, exist_ok=True)

    # NeMo ships this helper inside the package. Find the actual path.
    import nemo
    nemo_root = os.path.dirname(nemo.__file__)
    helper = os.path.join(nemo_root, "..", "scripts", "dataset_processing", "get_librispeech_data.py")
    if not os.path.exists(helper):
        # In some installs the scripts live under nemo_toolkit
        helper = os.path.join(nemo_root, "..", "..", "scripts", "dataset_processing", "get_librispeech_data.py")

    if not os.path.exists(helper):
        print("ERROR: get_librispeech_data.py not found.")
        print("Workaround: clone the NeMo repo and run:")
        print("    python <NeMo-repo>/scripts/dataset_processing/get_librispeech_data.py \\")
        print(f"        --data_root {args.data_root} --data_sets {args.data_sets}")
        sys.exit(1)

    cmd = [sys.executable, helper, "--data_root", args.data_root, "--data_sets", args.data_sets]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
