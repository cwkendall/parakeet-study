# Exercise 2 — extension problems

### 1. Shallow LM fusion in prefix beam (medium)
Extend `ctc_prefix_beam_search` to accept an `lm_score(prefix) -> float` callable. At each prefix extension, add `λ * lm_score(new_prefix) + α * len(new_prefix)` to the beam score. Test that adding a real (small) character n-gram LM trained on English text reduces character error rate on the toy classifier.

### 2. WFST decoding (hard)
Construct an explicit token-loop H FST and a small unigram L FST, compose them, and run Viterbi on the composed graph using your model's per-frame probabilities. Compare to greedy and prefix beam. This is the entire mental model behind Kaldi's chain models in 50 lines of Python.

### 3. Forced alignment (medium)
Use your forward variable to extract, for each frame, the *most likely* extended-target index. Compare the resulting per-token frame spans against the synthetic ground truth (which knows which frames came from which letter, since you generated them). Compute mean absolute alignment error.

### 4. InterCTC (medium)
Add auxiliary CTC heads after each intermediate conv layer of `TinyEncoder`. Sum the losses (weighted by depth — e.g. weight `i / num_layers`). Plot the convergence curve vs the original single-head training. InterCTC is a well-known trick (Lee & Watanabe 2021) that often speeds convergence by 1.5–2×.

### 5. Gradient-check your gradient (easy)
Numerically verify your analytical gradient by perturbing one logit at a time:
```python
def numerical_grad(f, logits, eps=1e-4):
    g = np.zeros_like(logits)
    for t in range(logits.shape[0]):
        for k in range(logits.shape[1]):
            logits[t,k] += eps; lp = f(logits); logits[t,k] -= 2*eps; lm = f(logits); logits[t,k] += eps
            g[t,k] = (lp - lm) / (2*eps)
    return g
```
You should get agreement with your analytical gradient to within `1e-4`.

### 6. Repeated-label torture test (easy)
Build cases where the target has three or more repeated labels in a row (e.g. `target=[1,1,1]`). Verify the forward variable's path-counting math agrees with brute-force enumeration of valid alignments for small `T`.

### 7. Reduce $|V|$ to characters and observe (easy)
Re-run the toy classifier with `VOCAB = "ab"` and only 4 two-letter words. The classifier should converge much faster. This is a sanity check that loss/gradient implementations scale correctly with vocabulary size.
