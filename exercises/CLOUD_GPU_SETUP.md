# Cloud GPU setup for exercises 4–6

You need an NVIDIA GPU with **≥16 GB VRAM** for these exercises (the small FastConformer fits in a T4; a 3090/4090/A4000 is more comfortable). Below are three concrete options, ordered by cost:

| Provider | Free tier | Paid hourly (24 GB-class) | Best for | Pain points |
|---|---|---|---|---|
| **Google Colab** | T4 16 GB, ~12 h/session, ~25 h/week | $10/mo Colab Pro = V100/A100 with priority | Ex 4 only, free, no setup | No SSH, no persistent disk (use Drive), session can drop mid-training |
| **Kaggle** | P100 16 GB or T4×2, 30 h/week, 12 h/session | n/a (free only) | Ex 4 free fallback if Colab full | Notebook-only, no SSH, weekly quota |
| **RunPod** | None | $0.34/h for RTX 4090 24 GB, $0.79/h A40 48 GB | Ex 4–6 end-to-end if you want SSH + persistent volume | Costs money. Budget ~$5–10 for Ex 4, ~$3 for Ex 5, ~$8 for Ex 6. |
| **Vast.ai** | None | ~$0.20/h for 3090 on spot (cheaper than RunPod, less reliable) | Cost-sensitive Ex 6 | Spot instances can be killed |
| **Lambda Labs** | None | $0.50/h H100 on-demand (cheapest H100 anywhere) | Overkill for these exercises | — |

**Recommendation:** Do Ex 4 on Colab free (T4 is enough for the 30M-param model and `train-clean-100`). Do Ex 5 and Ex 6 on RunPod RTX 4090 because they need warm checkpoints from the previous step and Colab's lack of persistent storage gets painful.

---

## Option A — Google Colab (free)

### 1. Create a Colab notebook

Go to <https://colab.research.google.com>, `File → New notebook`. Then `Runtime → Change runtime type → T4 GPU` (or A100 if you have Pro).

### 2. Mount Google Drive for persistent storage

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/parakeet-study
%cd /content/drive/MyDrive/parakeet-study
```

This survives session disconnects. Everything you save here is yours forever.

### 3. Install NeMo

```bash
!pip install -q "nemo_toolkit[asr]==2.0.0" "pytorch-lightning>=2.0" sentencepiece
!pip install -q librosa soundfile
```

NeMo installs cleanly on Colab — they pre-install matching CUDA/PyTorch. The whole install is ~3 minutes.

### 4. Clone the exercise

```bash
# If you've pushed your study pack to GitHub, clone it. Otherwise upload the exercise dir to Drive:
!cp -r /content/drive/MyDrive/parakeet-study/exercises/04-train-fastconformer-ctc .
%cd 04-train-fastconformer-ctc
```

### 5. Download LibriSpeech

LibriSpeech `train-clean-100` is 6 GB. Colab's `/content` (ephemeral) downloads at ~50 MB/s; Drive downloads at ~5 MB/s. **Download to `/content` (ephemeral), train, then save only the checkpoint to Drive:**

```bash
!python download_librispeech.py --data_root /content/librispeech --subsets train-clean-100,dev-clean,test-clean
```

~10 minutes for the download.

### 6. Train

Edit the config so checkpoints go to Drive, not ephemeral storage:

```yaml
# conf/fastconformer_ctc_small.yaml
exp_manager:
  exp_dir: /content/drive/MyDrive/parakeet-study/exp/fastconformer_ctc
```

Then:

```bash
!python train.py
```

If you hit a 12 h session limit before convergence, restart and resume:

```bash
!python train.py +init_from_ptl_ckpt=/content/drive/MyDrive/parakeet-study/exp/fastconformer_ctc/checkpoints/last.ckpt
```

### 7. Keep the session alive

Colab disconnects after ~90 min of idle. While training is actually running you're fine. If you ever need to leave a notebook open without training, this JS keepalive in the browser console helps:

```javascript
setInterval(() => {
  document.querySelector("colab-toolbar-button#connect")?.click();
}, 60000);
```

(Don't abuse this — Google does notice.)

---

## Option B — RunPod (cheap paid, the comfortable path for Ex 5–6)

RunPod gives you SSH + Jupyter + persistent volume. Total cost for Ex 4–6 done back-to-back: **~$15** on an RTX 4090.

### 1. Create an account

Sign up at <https://runpod.io>. Add $10 credit (minimum). No subscription.

### 2. Launch a pod

`Deploy → GPU Pods → RTX 4090 (24 GB)`. Settings that matter:

- **Template:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (NeMo's preferred CUDA).
- **Container disk:** 50 GB (ephemeral).
- **Volume:** create a **50 GB persistent volume** mounted at `/workspace`. This survives pod restarts — put your data and checkpoints here.
- **Spot vs on-demand:** spot is ~30% cheaper but can be killed. For Ex 4 (long training) prefer on-demand. For Ex 5/6 (short fine-tune) spot is fine.
- **Expose:** SSH (port 22) and Jupyter (port 8888) are exposed by default.

Click `Deploy`. Pod is ready in ~60s.

### 3. Connect

The Pod page shows `Connect → SSH over exposed TCP`. Copy the command, looks like:

```bash
ssh root@<host> -p <port> -i ~/.ssh/id_ed25519
```

Or click the Jupyter button to get a browser IDE.

### 4. Install & run

```bash
cd /workspace                                      # persistent volume
git clone <your-fork-of-parakeet-study>            # or scp the exercise dir up
cd parakeet-study/exercises/04-train-fastconformer-ctc

pip install -r requirements.txt                    # NeMo is ~3 GB, takes 5 min
python download_librispeech.py --data_root /workspace/librispeech
python train.py
```

### 5. Stop the pod when not training

RunPod bills per-second while running. **Always click `Stop` when you walk away.** Stopped pods cost nothing for compute, only ~$0.10/day for the persistent volume. Click `Resume` to come back — your `/workspace` is intact.

### 6. Copy artefacts down when finished

```bash
# from your laptop
scp -P <port> root@<host>:/workspace/parakeet-study/exercises/04-train-fastconformer-ctc/exp/.../checkpoints/last.ckpt ./
```

Or use `rclone` to push to Drive / S3.

---

## Option C — Kaggle (free, T4 or P100)

Kaggle gives 30 h/week of GPU time across all your notebooks, with sessions up to 12 h. P100 is a bit faster than T4 for training (16 GB HBM2 vs 16 GB GDDR6).

### 1. Create a kernel

<https://www.kaggle.com/code> → `New Notebook` → `Settings → Accelerator → GPU P100`.

### 2. Add a Kaggle Dataset for persistent storage

Kaggle doesn't have Drive-equivalent storage. Trick: after each training run, **save your checkpoint as a new dataset version**. Then mount that dataset as input on the next session.

```python
# In your training notebook, after training:
import shutil, os
shutil.copy('exp/.../last.ckpt', '/kaggle/working/last.ckpt')
# Then in the notebook UI: Output → Save → "Save & Run All" → versions it.
# Or use kaggle CLI:
#   kaggle datasets version -p /kaggle/working -m "epoch 50"
```

### 3. Install NeMo & train

Same as Colab from step 3 onward.

---

## Sanity check: how to know your setup actually works

Before starting Ex 4 properly, run this 60-second smoke test on whichever platform you chose:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

import nemo.collections.asr as nemo_asr
print(f"NeMo version: {nemo_asr.__file__}")

# Allocate a 4 GB tensor and run a matmul — confirms the GPU isn't gimped
x = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)
y = x @ x
torch.cuda.synchronize()
print("GPU compute works.")
```

If any of those fail, fix it before starting training — don't burn 4 hours discovering your VRAM is shared with the desktop.

## Cost worked example (RunPod RTX 4090, $0.34/h)

| Exercise | Wall time | Cost |
|---|---|---|
| Ex 4 (FastConformer-CTC, 50 epochs, train-clean-100) | ~14 h | ~$4.80 |
| Ex 5 (RNN-T fine-tune, 15 epochs from Ex 4 ckpt) | ~5 h | ~$1.70 |
| Ex 6 (TDT fine-tune, 20 epochs from Ex 5 ckpt) | ~8 h | ~$2.75 |
| **Total** | **~27 h** | **~$9.25** |

Add ~$2 for the persistent volume over a month. Round up: **budget $15 total** including some idle time and re-runs.
