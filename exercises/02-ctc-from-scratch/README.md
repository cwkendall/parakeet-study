# Exercise 2 — CTC from scratch

**Time budget:** ~1 day.
**Prerequisites:** chapter 21 of the deep-dive (`../../index.html#ch-ctc`) and the Awni Hannun *Distill* article (linked at the bottom of that chapter).
**Builds:** the CTC loss function and the two most common CTC decoders.

By the end of this exercise you will have written, in pure NumPy:

1. The **extended target sequence** construction (interleaving blanks).
2. The **forward** variable `α(t, s)`.
3. The **backward** variable `β(t, s)`.
4. The **CTC loss** and its analytical gradient.
5. The **greedy CTC decoder** (argmax + collapse).
6. The **prefix beam search** decoder (the right way to decode CTC).

…and verified each piece against PyTorch's `torch.nn.functional.ctc_loss`. You will also use your loss to train a tiny CNN to align a sequence of "phoneme" frames to a target string — a self-contained miniature speech recogniser.

## Why this matters

CTC is the single most important loss function in the history of end-to-end ASR. Understanding the forward-backward derivation is non-negotiable for anyone working on ASR — it's the same dynamic-programming pattern that RNN-T and TDT (exercises 3 and 6) generalise. The Hannun *Distill* article is the canonical visual reference; this exercise is the matching coding exercise.

## The tutorial

Open `starter.py`. There are six **TODO blocks**.

### Step 1 — the extended target (10 minutes)

Given a target sequence like `"CAT"` (encoded as integer token IDs `[3, 1, 20]`), build the *extended target* by inserting blanks at the start, end, and between every pair of tokens:

```
target:           [3, 1, 20]              len U = 3
extended target:  [B, 3, B, 1, B, 20, B]  len S = 2U + 1 = 7
```

Where `B` is your designated blank index (we'll use the last vocabulary index by convention).

Why? Because the CTC alignment paths run on the *extended* target, not the raw target. The blanks between adjacent tokens are what allow alignment paths to differentiate consecutive identical tokens (e.g. `"HELLO"` needs the middle blank between the two L's so collapse doesn't turn it into "HELO").

Implement `extend_target(target, blank)`. Test:

```bash
pytest test_ctc.py::test_extend_target -v
```

### Step 2 — the forward variable (45 minutes)

The forward variable `α(t, s)` is the total probability of all valid alignment paths that reach `(t, s)` in the lattice — frame `t`, extended-target position `s`.

The recursion (read chapter 21 if this isn't natural yet):

```
α(t, s) = (α(t-1, s) + α(t-1, s-1)) * y[t, ext[s]]                                    if ext[s] == B or ext[s] == ext[s-2]
α(t, s) = (α(t-1, s) + α(t-1, s-1) + α(t-1, s-2)) * y[t, ext[s]]                       otherwise
```

with initial conditions:
- `α(0, 0) = y[0, B]`            (start in the leading blank)
- `α(0, 1) = y[0, ext[1]]`       (or start on the first label)
- `α(0, s) = 0` for `s >= 2`     (you can't skip ahead at t=0)

`y` here is the **probability matrix**: `y[t, k] = softmax(logits[t])[k]`. Shape: `[T, V+1]` where the +1 is the blank.

Implement `ctc_forward(log_probs, target, blank)`. Use **log-space** arithmetic — alignment probabilities go to zero very quickly in linear space. You'll need a numerically stable `logsumexp`:

```python
def logsumexp(*xs):
    m = max(xs)
    return m + np.log(sum(np.exp(x - m) for x in xs))
```

Test:

```bash
pytest test_ctc.py::test_forward_values    -v
pytest test_ctc.py::test_forward_endpoint -v
```

The first test pins specific numerical values of `α` at known cells for a tiny hand-checkable example. The second checks that `logsumexp(α(T-1, 2U-1), α(T-1, 2U))` (the two valid end states) equals the same loss PyTorch computes.

### Step 3 — the backward variable (30 minutes)

The backward variable `β(t, s)` is the total probability of all paths from `(t, s)` to the end. Same idea, recursing from the end:

```
β(t, s) = (β(t+1, s) + β(t+1, s+1)) * y[t+1, ext[s+1]?]                 # if blank or repeat constraint applies
β(t, s) = (β(t+1, s) + β(t+1, s+1) + β(t+1, s+2)) * ...                  # otherwise
```

with `β(T-1, 2U) = 1` and `β(T-1, 2U-1) = 1`.

Implement `ctc_backward(log_probs, target, blank)`.

Test:

```bash
pytest test_ctc.py::test_backward_endpoint -v
```

You should find that `α(T-1, 2U) + α(T-1, 2U-1) == β(0, 0) + β(0, 1)` (modulo floating point) — both compute the total path probability from two directions.

### Step 4 — loss and gradient (45 minutes)

The CTC loss is just `-log P(y|x) = -logsumexp(α(T-1, 2U-1), α(T-1, 2U))`.

The gradient is where the forward and backward variables both pay off. For each `(t, s)` cell, the *occupancy* — the probability that paths pass through it — is:

```
γ(t, s) = α(t, s) * β(t, s) / P(y|x)
```

The gradient of the loss with respect to the *log-softmax inputs* (i.e. the encoder logits) is:

```
dL/dlogits[t, k] = y[t, k] - sum over s such that ext[s] == k of γ(t, s)
```

This is the same form as cross-entropy: "softmax output minus one-hot target", except the "target" here is the marginal occupancy of label `k` at frame `t`, computed over all valid alignments.

Implement `ctc_loss_and_grad(logits, target, blank)`. Test:

```bash
pytest test_ctc.py::test_loss_matches_pytorch     -v
pytest test_ctc.py::test_gradient_matches_pytorch -v
```

PyTorch's `torch.nn.functional.ctc_loss` and our loss should agree to about `1e-5`. The gradient comparison checks that your analytical gradient matches PyTorch's autograd gradient — if it doesn't, you have a bug.

### Step 5 — greedy decoder (15 minutes)

The simplest CTC decoder: at every frame `t`, take `argmax(y[t])`. Then collapse the result: remove repeated tokens, then remove blanks.

```python
def ctc_greedy_decode(log_probs, blank):
    raw = np.argmax(log_probs, axis=1)              # [T]
    # collapse repeats
    collapsed = [raw[0]]
    for c in raw[1:]:
        if c != collapsed[-1]:
            collapsed.append(c)
    # remove blanks
    return [c for c in collapsed if c != blank]
```

Test:

```bash
pytest test_ctc.py::test_greedy_decode -v
```

### Step 6 — prefix beam search (60 minutes)

The greedy decoder is fast but blind to the conditional independence pitfall: it can pick a path that, when collapsed, is not the highest-probability *transcript*. Prefix beam search fixes this by tracking sets of *prefixes* (collapsed token strings) rather than alignment paths.

The key trick: for each prefix you track **two probabilities** — one ending in a blank (`pb`) and one ending in a non-blank (`pnb`). This lets you correctly handle the merge:

- Adding a blank to either prefix leaves the prefix unchanged but updates `pb`.
- Adding a repeat of the last non-blank requires going through a blank first, so it only contributes to `pnb` from `pb`.
- Adding a fresh token always extends the prefix.

Pseudocode (Hannun's *Distill* article has the full version):

```
beam = { (): (pb=1.0, pnb=0.0) }
for t in range(T):
    new_beam = {}
    for prefix, (pb, pnb) in beam.items():
        for s in range(V+1):
            p = y[t, s]
            if s == BLANK:
                # extend ending in blank; prefix stays the same
                new_pb, new_pnb = new_beam.get(prefix, (0, 0))
                new_pb += (pb + pnb) * p
                new_beam[prefix] = (new_pb, new_pnb)
            else:
                ext_prefix = prefix + (s,)
                new_pb_ext, new_pnb_ext = new_beam.get(ext_prefix, (0, 0))
                if len(prefix) > 0 and s == prefix[-1]:
                    # repeated label: requires a blank in between
                    new_pnb_ext += pb * p
                    # also: staying at the same prefix but updating pnb
                    new_pb_same, new_pnb_same = new_beam.get(prefix, (0, 0))
                    new_pnb_same += pnb * p
                    new_beam[prefix] = (new_pb_same, new_pnb_same)
                else:
                    new_pnb_ext += (pb + pnb) * p
                new_beam[ext_prefix] = (new_pb_ext, new_pnb_ext)
    # keep top-K
    beam = dict(sorted(new_beam.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:beam_size])
return max(beam.items(), key=lambda kv: kv[1][0] + kv[1][1])[0]
```

Implement `ctc_prefix_beam_search(log_probs, blank, beam_size=10)`. Test:

```bash
pytest test_ctc.py::test_beam_search_beats_greedy -v
```

The test constructs a probability matrix where greedy makes a mistake (because the highest-prob path doesn't correspond to the highest-prob transcript when paths merge) and verifies beam search gets it right.

### Step 7 (optional) — train a tiny classifier with your loss (3-4 hours)

`train_toy_classifier.py` is a runnable script that:

1. Generates synthetic "phoneme" sequences — 16-dimensional acoustic features, each sequence labelled with a short alphabetic transcript.
2. Builds a tiny CNN (3 conv layers, ~50K params).
3. Trains it with your `ctc_loss_and_grad` (no PyTorch loss).
4. Reports loss per epoch and decoded examples.

You shouldn't need to modify it — running it is the reward for completing the exercise. Expected output:

```
Epoch 0: loss=12.4, sample decode: 'qqq' (truth: 'cat')
Epoch 5: loss=4.1,  sample decode: 'cat' (truth: 'cat')
Epoch 20: loss=0.8, sample decode: 'cat' (truth: 'cat')
```

## Running everything

```bash
pytest test_ctc.py -v          # all CTC unit tests
python train_toy_classifier.py # optional: see your loss training a model
```

## Reflection

1. Why is `pb`/`pnb` necessary in prefix beam search? Construct a 2-step example where merging them would give the wrong answer.
2. What happens to the greedy decoder if you set the blank probability to a constant 0.99 at every frame? What does this tell you about how CTC models tend to behave when undertrained?
3. The forward-backward is $O(T \cdot S \cdot |V|)$ in the *worst* case, but actually $O(T \cdot S)$ for a single target sequence. Why? (Hint: which symbols at frame `t` get nonzero `α` updates?)

## Extension problems

See [`notes.md`](./notes.md):

- Implement an n-gram LM shallow-fusion variant of prefix beam search.
- Implement WFST-style decoding: build a token-loop FST and a small LM FST, compose them, and run Viterbi.
- Use your forward variable to extract a *forced alignment*: the most likely (path-)sequence of token-frame assignments.
- Implement "InterCTC" — adding auxiliary CTC heads at intermediate encoder layers — and quantify the convergence speedup on the toy task.
