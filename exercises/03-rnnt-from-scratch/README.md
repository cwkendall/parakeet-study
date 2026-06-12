# Exercise 3 — RNN-Transducer from scratch

**Time budget:** ~1 day.
**Prerequisites:** chapter 22 (`../../index.html#ch-rnnt`) and exercise 2 done.
**Builds:** the RNN-T forward-backward, loss, and greedy decoder.

By the end of this exercise you will have written, in pure NumPy:

1. The RNN-T **forward** variable `α(t, u)` over the 2D lattice.
2. The **backward** variable `β(t, u)`.
3. The **loss** `-log α(T, U) - log P(∅ | T, U)` and **gradient** wrt the joint-network output.
4. A **greedy decoder** matching the canonical Graves-2012 algorithm.

…and verified each against PyTorch's [`torchaudio.functional.rnnt_loss`](https://pytorch.org/audio/main/generated/torchaudio.functional.rnnt_loss.html). The forward-backward derivation here is the **same dynamic-programming pattern as CTC**, just with a second axis added — recognising that should make exercise 6 (TDT) feel like an incremental extension rather than a new beast.

## Why this matters

CTC has one axis (time). RNN-T has two (time × output position). All modern streaming/transducer ASR systems live on this 2D lattice; once you've coded the forward-backward by hand you'll never be confused by `(t, u)` indexing in NeMo source code again. Exercise 6's TDT is one more axis of generalisation on top of this.

## The tutorial

Open `starter.py`. There are five **TODO blocks**.

### Step 1 — set up the joint distribution (10 minutes)

You won't actually train an encoder or prediction net here — that comes in exercises 4–6. Instead, the tests **feed you a precomputed joint tensor**:

```
joint_log_probs: shape [T, U+1, V+1]
    joint_log_probs[t, u, v] = log P(emit v | encoder frame t, prediction state u)
```

`v == BLANK` is the blank index; `v != BLANK` is a vocabulary token.

The simplification: in real RNN-T training the joint network produces this tensor on the fly from `[encoder_out, prednet_out]`. For this exercise we hand it to you as a tensor — that isolates the forward-backward from the surrounding networks.

Implement `extract_blank_and_label_log_probs(joint_log_probs, target, blank)` which returns two arrays:

- `log_p_blank[t, u]`  = `joint_log_probs[t, u, blank]`
- `log_p_target[t, u]` = `joint_log_probs[t, u, target[u]]`  (only valid for `u < U`)

These are the only two probabilities the forward-backward needs at each lattice cell.

### Step 2 — forward variable (45 minutes)

The recursion (see chapter 22):

```
α(t, u) = α(t-1, u)   * P(∅      | t-1, u)        (came from the left: blank advances time)
        + α(t, u-1)   * P(y_u    | t, u-1)         (came from above: token advances target)
```

with `α(0, 0) = 1`.

In log space:

```
log_α(t, u) = logsumexp(
    log_α(t-1, u)   + log P(∅   | t-1, u),
    log_α(t, u-1)   + log P(y_u | t, u-1),
)
```

Edge cases:
- `log_α(0, 0) = 0`     (probability 1)
- `log_α(t, 0)` for `t > 0` = `log_α(t-1, 0) + log P(∅ | t-1, 0)`  (only one way to get here)
- `log_α(0, u)` for `u > 0` = `log_α(0, u-1) + log P(y_u | 0, u-1)`

Implement `rnnt_forward(log_p_blank, log_p_target)` returning a `[T+1, U+1]` log-α tensor. (The +1 on T is conventional; many treatments use `[T, U+1]`. We follow the convention where the loss is `log_α[T, U] + log P(∅ | T-1, U)`.)

Test:

```bash
pytest test_rnnt.py::test_forward_endpoint -v
```

This checks your `α(T, U)` against `torchaudio.functional.rnnt_loss` for random inputs.

### Step 3 — backward variable (30 minutes)

The recursion runs in reverse:

```
log_β(t, u) = logsumexp(
    log P(∅   | t, u) + log_β(t+1, u),
    log P(y_{u+1} | t, u) + log_β(t, u+1),   # only if u < U
)
```

Initial condition: `log_β(T, U) = 0`.

Verify that `α + β` is constant along any anti-diagonal: this is the consistency check the test runs.

### Step 4 — loss and gradient (60 minutes)

Loss:

```
loss = -log P(y | x) = -(log_α[T-1, U] + log P(∅ | T-1, U))
```

Gradient with respect to the joint output (analogous to CTC's case):

```
For each (t, u, v):
    γ(t, u, v) = α(t, u) * β(t, u) * P(v | t, u) * (probability that paths take this transition)
```

The clean derivation (Graves 2012 §4 has the full thing):

```
dL/d_joint_log_probs[t, u, blank] = -exp(log_α[t, u] + log P(∅|t, u) + log_β[t+1, u] - log_P)    if t < T
                                  = -exp(log_α[T-1, U] + log P(∅|T-1, U) - log_P)               if (t, u) = (T-1, U)
dL/d_joint_log_probs[t, u, target[u]] = -exp(log_α[t, u] + log P(y_{u+1}|t, u) + log_β[t, u+1] - log_P)   if u < U
dL/d_joint_log_probs[t, u, v] = 0 for all other v
```

The negative sign comes from `loss = -log P`.

These are *log-probability* gradients. If you want gradients wrt the *logits* (the pre-softmax inputs), you have to push through the softmax, which is the standard "softmax(x) - one_hot" form.

Implement `rnnt_loss_and_grad(joint_log_probs, target, blank)` returning `(loss, grad_wrt_log_probs)`.

Test:

```bash
pytest test_rnnt.py::test_loss_matches_torchaudio       -v
pytest test_rnnt.py::test_gradient_matches_torchaudio   -v
```

### Step 5 — greedy decoder (20 minutes)

Algorithm from Graves 2012 §4 (matches NeMo's basic transducer greedy decoder):

```
t, u = 0, 0; hyp = []
while t < T:
    v = argmax(joint_log_probs[t, u])
    if v == BLANK:
        t += 1
    else:
        hyp.append(v); u += 1
        if symbols_at_this_t > MAX_SYMBOLS:    # the same guard as TDT
            t += 1
return hyp
```

Implement `rnnt_greedy_decode(joint_log_probs, blank, max_symbols_per_step=10)`. Note that this version assumes the joint tensor is precomputed; the real algorithm calls `prednet_step()` after each emission to update `u` and `g_u`, but since we're testing the lattice math, not the prediction net, we collapse that to a tensor lookup.

Test:

```bash
pytest test_rnnt.py::test_greedy_decode -v
```

## Running everything

```bash
pytest test_rnnt.py -v
```

## Reflection

1. Why does the forward variable not need to track *which* token the prediction network is on? (Hint: the joint tensor already conditions on `u`.)
2. RNN-T is $O(T \cdot U)$ per training example. CTC is $O(T \cdot S) = O(T \cdot U)$. Why is RNN-T memory-hungrier in practice? (Hint: the joint tensor.)
3. Greedy RNN-T can in principle emit an unbounded number of tokens at one frame. The `MAX_SYMBOLS` guard prevents this in practice. What would be the symptom in a real ASR system if this guard didn't exist? (Answer: hallucinated repetition. Search NeMo for `max_symbols` and read why.)

## Extension problems

See [`notes.md`](./notes.md):

- Implement the RNN-T beam-search variants (TSD, ALSD, MAES).
- Verify your gradient by finite differences in addition to autograd cross-check.
- Implement *pruned* RNN-T training (k2-style): only update lattice cells inside a band around the diagonal alignment, big memory savings.
- Add a prediction network (one-layer LSTM) and a small joint network, and train end-to-end on the synthetic data from exercise 2.
