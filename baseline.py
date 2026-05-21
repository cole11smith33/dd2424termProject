import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import torch
import copy
import json

# ── Reproducibility ───────────────────────────────────────────────────────────
seed = 42
rng  = np.random.default_rng(seed)

# ── Load data ─────────────────────────────────────────────────────────────────
current_file = Path(__file__).resolve()
book_dir     = current_file.parent / "data"
book_fname   = book_dir / "train.txt"

with open(book_fname, "r") as fid:
    book_data = fid.read()

unique_chars = sorted(list(set(book_data)))
K = len(unique_chars)

char_to_ind = {ch: i  for i, ch in enumerate(unique_chars)}
ind_to_char = {i:  ch for i, ch in enumerate(unique_chars)}

# ── Train / Val / Test split (80 / 10 / 10 by character position) ─────────────
n         = len(book_data)
train_end = int(0.80 * n)
val_end   = int(0.90 * n)

train_data = book_data[:train_end]
val_data   = book_data[train_end:val_end]
test_data  = book_data[val_end:]

print(f"Vocabulary size : {K}")
print(f"Total chars     : {n:,}")
print(f"Train           : {len(train_data):,}  ({len(train_data)/n:.1%})")
print(f"Validation      : {len(val_data):,}   ({len(val_data)/n:.1%})")
print(f"Test            : {len(test_data):,}   ({len(test_data)/n:.1%})")

# ── Hyperparameters ───────────────────────────────────────────────────────────
m          = 100    # hidden state dimensionality
seq_length = 25     # training sequence length
eta        = 1e-3   # Adam learning rate
beta1      = 0.9
beta2      = 0.999
eps        = 1e-8

# ── Model init ────────────────────────────────────────────────────────────────
def init_rnn(m, K, rng):
    return {
        'b': np.zeros((m, 1)),
        'c': np.zeros((K, 1)),
        'U': (1/np.sqrt(2*K)) * rng.standard_normal((m, K)),
        'W': (1/np.sqrt(2*m)) * rng.standard_normal((m, m)),
        'V': (1/np.sqrt(m))   * rng.standard_normal((K, m)),
    }

RNN = init_rnn(m, K, rng)

# ── Helpers ───────────────────────────────────────────────────────────────────
def chars_to_onehot(chars):
    T = len(chars)
    X = np.zeros((K, T))
    for t, ch in enumerate(chars):
        X[char_to_ind[ch], t] = 1.0
    return X


def synthesize(model, initial_hidden_state, start_input, sequence_length):
    hidden_state  = initial_hidden_state.copy()
    current_input = start_input.copy()
    generated_indices = []
    vocab_size = model['U'].shape[1]

    for _ in range(sequence_length):
        hidden_state = np.tanh(model['W'] @ hidden_state + model['U'] @ current_input + model['b'])
        logits       = model['V'] @ hidden_state + model['c']
        logits      -= np.max(logits)
        probs        = np.exp(logits) / np.sum(np.exp(logits))
        cumprobs     = np.cumsum(probs)
        idx          = int(np.where(cumprobs > rng.uniform())[0][0])
        generated_indices.append(idx)
        current_input = np.zeros((vocab_size, 1))
        current_input[idx, 0] = 1.0

    return generated_indices


def indices_to_string(indices):
    return "".join(ind_to_char[i] for i in indices)


def compute_loss_on_split(model, data, seq_length):
    """Average cross-entropy loss over an entire data split."""
    total_loss = 0.0
    n_seqs     = 0
    hprev      = np.zeros((model['W'].shape[0], 1))

    for e in range(0, len(data) - seq_length - 1, seq_length):
        X = chars_to_onehot(data[e : e + seq_length])
        Y = chars_to_onehot(data[e + 1 : e + seq_length + 1])
        loss, _, hprev = forward(model, X, Y, hprev)
        total_loss += loss
        n_seqs += 1

    return total_loss / n_seqs if n_seqs > 0 else float('nan')

# ── Forward / Backward ────────────────────────────────────────────────────────
def forward(model, inputs, targets, initial_hidden_state):
    T         = inputs.shape[1]
    hidden_dim = model['W'].shape[0]
    vocab_size = model['V'].shape[0]

    H  = np.zeros((hidden_dim, T + 1))
    A  = np.zeros((hidden_dim, T))
    P  = np.zeros((vocab_size, T))
    H[:, [0]] = initial_hidden_state
    total_loss = 0.0

    for t in range(T):
        A[:, t:t+1]   = model['W'] @ H[:, t:t+1] + model['U'] @ inputs[:, t:t+1] + model['b']
        H[:, t+1:t+2] = np.tanh(A[:, t:t+1])
        logits        = model['V'] @ H[:, t+1:t+2] + model['c']
        logits       -= np.max(logits)
        p             = np.exp(logits) / np.sum(np.exp(logits))
        P[:, t:t+1]   = p
        total_loss   -= np.log(p[np.argmax(targets[:, t]), 0])

    cache = {'inputs': inputs, 'targets': targets, 'H': H, 'A': A, 'P': P}
    return total_loss / T, cache, H[:, [T]]


def backward(model, cache):
    inputs  = cache['inputs']
    targets = cache['targets']
    H       = cache['H']
    P       = cache['P']
    T       = inputs.shape[1]

    grads          = {k: np.zeros_like(v) for k, v in model.items()}
    dh_next        = np.zeros((model['W'].shape[0], 1))

    for t in reversed(range(T)):
        dlogits         = P[:, t:t+1] - targets[:, t:t+1]
        grads['V']     += dlogits @ H[:, t+1:t+2].T
        grads['c']     += dlogits
        dh              = model['V'].T @ dlogits + dh_next
        da              = dh * (1 - H[:, t+1:t+2]**2)
        grads['W']     += da @ H[:, t:t+1].T
        grads['U']     += da @ inputs[:, t:t+1].T
        grads['b']     += da
        dh_next         = model['W'].T @ da

    for k in grads:
        grads[k] /= T
        grads[k]  = np.clip(grads[k], -5, 5)

    return grads

# ── Gradient check ────────────────────────────────────────────────────────────
def ComputeGradsWithTorch_col(X, y_indices, h0, RNN):
    tau = X.shape[1]
    Xt  = torch.from_numpy(X)
    ht  = torch.from_numpy(h0)
    net = {k: torch.tensor(v, requires_grad=True) for k, v in RNN.items()}
    tanh, softmax = torch.nn.Tanh(), torch.nn.Softmax(dim=0)
    Hs = torch.empty(h0.shape[0], tau, dtype=torch.float64)
    hprev = ht
    for t in range(tau):
        a = net['W'] @ hprev + net['U'] @ Xt[:, t:t+1] + net['b']
        hprev = tanh(a)
        Hs[:, t:t+1] = hprev
    Os = torch.matmul(net['V'], Hs) + net['c']
    P  = softmax(Os)
    torch.mean(-torch.log(P[y_indices, np.arange(tau)])).backward()
    return {k: net[k].grad.numpy() for k in RNN}


print("\nGradient check (m=10, seq_length=25)")
m_chk   = 10
RNN_chk = init_rnn(m_chk, K, np.random.default_rng(0))
h0_chk  = np.zeros((m_chk, 1))
X_chk   = chars_to_onehot(book_data[:seq_length])
Y_oh    = chars_to_onehot(book_data[1:seq_length+1])
y_idx   = np.argmax(Y_oh, axis=0)

_, cache_chk, _ = forward(RNN_chk, X_chk, Y_oh, h0_chk)
my_grads    = backward(RNN_chk, cache_chk)
torch_grads = ComputeGradsWithTorch_col(X_chk, y_idx, h0_chk, RNN_chk)
eps_rel = 1e-20
for key in ['W', 'U', 'V', 'b', 'c']:
    rel = np.abs(my_grads[key] - torch_grads[key]) / np.maximum(eps_rel, np.abs(my_grads[key]) + np.abs(torch_grads[key]))
    print(f"  {key}  max relative error: {rel.max():.2e}")

# ── Adam ──────────────────────────────────────────────────────────────────────
def init_adam(RNN):
    return ({k: np.zeros_like(v) for k, v in RNN.items()},
            {k: np.zeros_like(v) for k, v in RNN.items()})


def adam_update(RNN, grads, m_adam, v_adam, t_adam, eta, beta1, beta2, eps):
    for k in RNN:
        m_adam[k] = beta1 * m_adam[k] + (1 - beta1) * grads[k]
        v_adam[k] = beta2 * v_adam[k] + (1 - beta2) * grads[k]**2
        m_hat = m_adam[k] / (1 - beta1**t_adam)
        v_hat = v_adam[k] / (1 - beta2**t_adam)
        RNN[k] -= eta * m_hat / (np.sqrt(v_hat) + eps)

# ── Training loop ─────────────────────────────────────────────────────────────
def train(RNN, train_data, val_data, n_epochs=3,
          print_every=100, synth_every=1000, synth_len=200,
          val_every=500):
    """
    Returns
    -------
    results : dict with keys
        'train_smooth'  : list of (step, smooth_loss)
        'val_loss'      : list of (step, val_loss)   – measured every val_every steps
        'epoch_val_loss': list of (epoch, val_loss)  – measured at epoch end
        'final_test_placeholder': None  (filled externally after training)
    """
    m_adam, v_adam = init_adam(RNN)
    t_adam     = 0
    step       = 0
    smooth_loss = None

    train_smooth   = []   # (step, smooth_loss)
    val_curve      = []   # (step, val_loss)
    epoch_val      = []   # (epoch, val_loss)

    best_loss = float('inf')
    best_RNN  = None

    for epoch in range(1, n_epochs + 1):
        e     = 0
        hprev = np.zeros((m, 1))

        while e <= len(train_data) - seq_length - 1:
            X = chars_to_onehot(train_data[e : e + seq_length])
            Y = chars_to_onehot(train_data[e + 1 : e + seq_length + 1])

            loss, cache, hprev = forward(RNN, X, Y, hprev)
            grads = backward(RNN, cache)

            t_adam += 1
            adam_update(RNN, grads, m_adam, v_adam, t_adam, eta, beta1, beta2, eps)

            smooth_loss = loss if smooth_loss is None else 0.999 * smooth_loss + 0.001 * loss

            if step == 0 or (step + 1) % print_every == 0:
                print(f"epoch {epoch}  step {step+1:6d}  smooth_loss={smooth_loss:.4f}")
                train_smooth.append((step + 1, smooth_loss))

            # ── Periodic validation loss ──────────────────────────────────────
            if step == 0 or (step + 1) % val_every == 0:
                v_loss = compute_loss_on_split(RNN, val_data, seq_length)
                val_curve.append((step + 1, v_loss))
                print(f"  [val loss @ step {step+1}] {v_loss:.4f}")

            if smooth_loss < best_loss:
                best_loss = smooth_loss
                best_RNN  = copy.deepcopy(RNN)

            if step == 0 or (step + 1) % synth_every == 0:
                x0_ = X[:, 0:1].copy()
                idx = synthesize(RNN, hprev, x0_, synth_len)
                print(f"\n  [synth at step {step+1}]\n{indices_to_string(idx)}\n")

            e    += seq_length
            step += 1

        # ── Epoch-end validation ──────────────────────────────────────────────
        v_loss_epoch = compute_loss_on_split(RNN, val_data, seq_length)
        epoch_val.append((epoch, v_loss_epoch))
        print(f"\n── End of epoch {epoch}  val_loss={v_loss_epoch:.4f} ──\n")
        hprev = np.zeros((m, 1))

    results = {
        'model_name':         'Vanilla RNN',
        'hidden_size':        m,
        'train_smooth':       train_smooth,
        'val_curve':          val_curve,
        'epoch_val_loss':     epoch_val,
        'final_test_loss':    None,   # filled below
        'best_val_loss':      min(v for _, v in val_curve),
    }
    return results, best_RNN


# ── Run training ──────────────────────────────────────────────────────────────
results, best_RNN = train(
    RNN, train_data, val_data,
    n_epochs=3,
    print_every=100,
    synth_every=1000,
    synth_len=200,
    val_every=500,
)

# ── Final test loss ───────────────────────────────────────────────────────────
test_loss = compute_loss_on_split(best_RNN, test_data, seq_length)
results['final_test_loss'] = test_loss
print(f"\nFinal test loss (best checkpoint): {test_loss:.4f}")
print(f"Final test perplexity            : {np.exp(test_loss):.2f}")

# ── Save results for later comparison ─────────────────────────────────────────
# Convert tuples → lists for JSON serialisation
def serialise(r):
    out = {}
    for k, v in r.items():
        if isinstance(v, list) and v and isinstance(v[0], tuple):
            out[k] = [list(x) for x in v]
        else:
            out[k] = v
    return out

results_path = book_dir / "rnn_results.json"
with open(results_path, "w") as f:
    json.dump(serialise(results), f, indent=2)
print(f"Results saved to {results_path}")

# ── Generate text from best model ─────────────────────────────────────────────
h0_best = np.zeros((m, 1))
x0_best = np.zeros((K, 1))
x0_best[char_to_ind['.'], 0] = 1.0
best_text = indices_to_string(synthesize(best_RNN, h0_best, x0_best, 1000))
print("\nText from best model:\n")
print(best_text)
with open(book_dir / "best_model_sample.txt", "w") as f:
    f.write(best_text)

# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON PLOTTING
# ══════════════════════════════════════════════════════════════════════════════
# How to use:
#   After training your LSTM models, build a list `all_results` where each
#   entry is a dict in the same format returned by train() above.
#   Then call plot_comparison(all_results).
#
# Minimal LSTM result dict shape:
#   {
#     'model_name':     'LSTM 1-layer',
#     'hidden_size':    128,
#     'train_smooth':   [(step, loss), ...],
#     'val_curve':      [(step, loss), ...],
#     'epoch_val_loss': [(epoch, loss), ...],
#     'final_test_loss': 2.34,
#     'best_val_loss':   2.10,
#   }
# ══════════════════════════════════════════════════════════════════════════════

COLORS = ['#3266ad', '#d85a30', '#1d9e75', '#7f77dd', '#ba7517', '#993556']
STYLES = ['-', '--', '-.', ':', '-', '--']

def plot_comparison(all_results, save_path=None):
    """
    all_results : list of result dicts (one per model).
    Produces a 2×2 figure:
      [0,0] Training smooth loss curves
      [0,1] Validation loss curves (periodic)
      [1,0] Epoch-end validation loss
      [1,1] Final test loss + perplexity bar chart
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Model comparison — Vanilla RNN vs LSTM variants", fontsize=14, y=0.98)
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.30)

    ax_train = fig.add_subplot(gs[0, 0])
    ax_val   = fig.add_subplot(gs[0, 1])
    ax_epoch = fig.add_subplot(gs[1, 0])
    ax_bar   = fig.add_subplot(gs[1, 1])

    # ── (0,0) Training smooth loss ────────────────────────────────────────────
    for i, r in enumerate(all_results):
        if not r['train_smooth']:
            continue
        steps, losses = zip(*r['train_smooth'])
        ax_train.plot(steps, losses,
                      color=COLORS[i % len(COLORS)],
                      linestyle=STYLES[i % len(STYLES)],
                      linewidth=1.6,
                      label=r['model_name'])
    ax_train.set_title("Training smooth loss", fontsize=11)
    ax_train.set_xlabel("Update step")
    ax_train.set_ylabel("Smooth loss")
    ax_train.legend(fontsize=8)
    ax_train.grid(True, alpha=0.3)

    # ── (0,1) Validation loss curve ───────────────────────────────────────────
    for i, r in enumerate(all_results):
        if not r['val_curve']:
            continue
        steps, losses = zip(*r['val_curve'])
        ax_val.plot(steps, losses,
                    color=COLORS[i % len(COLORS)],
                    linestyle=STYLES[i % len(STYLES)],
                    linewidth=1.6,
                    label=r['model_name'])
    ax_val.set_title("Validation loss (per eval step)", fontsize=11)
    ax_val.set_xlabel("Update step")
    ax_val.set_ylabel("Validation loss")
    ax_val.legend(fontsize=8)
    ax_val.grid(True, alpha=0.3)

    # ── (1,0) Epoch-end validation loss ───────────────────────────────────────
    for i, r in enumerate(all_results):
        if not r['epoch_val_loss']:
            continue
        epochs, losses = zip(*r['epoch_val_loss'])
        ax_epoch.plot(epochs, losses,
                      marker='o', markersize=6,
                      color=COLORS[i % len(COLORS)],
                      linestyle=STYLES[i % len(STYLES)],
                      linewidth=1.6,
                      label=r['model_name'])
    ax_epoch.set_title("Epoch-end validation loss", fontsize=11)
    ax_epoch.set_xlabel("Epoch")
    ax_epoch.set_ylabel("Validation loss")
    ax_epoch.set_xticks(range(1, max(len(r['epoch_val_loss']) for r in all_results) + 1))
    ax_epoch.legend(fontsize=8)
    ax_epoch.grid(True, alpha=0.3)

    # ── (1,1) Final test loss + perplexity bar chart ──────────────────────────
    names       = [r['model_name']     for r in all_results]
    test_losses = [r['final_test_loss'] if r['final_test_loss'] is not None else float('nan')
                   for r in all_results]
    perplexities = [np.exp(tl) if not np.isnan(tl) else float('nan') for tl in test_losses]

    x      = np.arange(len(names))
    width  = 0.38
    bars1  = ax_bar.bar(x - width/2, test_losses, width,
                        color=[COLORS[i % len(COLORS)] for i in range(len(names))],
                        alpha=0.85, label='Test loss')

    ax_bar2 = ax_bar.twinx()
    bars2   = ax_bar2.bar(x + width/2, perplexities, width,
                          color=[COLORS[i % len(COLORS)] for i in range(len(names))],
                          alpha=0.40, hatch='//', label='Perplexity')

    # Value labels on bars
    for bar, val in zip(bars1, test_losses):
        if not np.isnan(val):
            ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, perplexities):
        if not np.isnan(val):
            ax_bar2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                         f"{val:.1f}", ha='center', va='bottom', fontsize=8)

    ax_bar.set_title("Final test loss & perplexity", fontsize=11)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names, rotation=15, ha='right', fontsize=8)
    ax_bar.set_ylabel("Test loss")
    ax_bar2.set_ylabel("Perplexity")
    ax_bar.grid(True, axis='y', alpha=0.3)

    lines = [plt.Line2D([0], [0], color='gray', alpha=0.85, linewidth=6, label='Test loss'),
             plt.Line2D([0], [0], color='gray', alpha=0.40, linewidth=6, label='Perplexity')]
    ax_bar.legend(handles=lines, fontsize=8, loc='upper right')

    # ── Save / show ───────────────────────────────────────────────────────────
    if save_path is None:
        save_path = book_dir / "model_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Comparison figure saved to {save_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "="*68)
    print(f"{'Model':<22} {'Hidden':>6} {'Best val':>10} {'Test loss':>10} {'Perplexity':>11}")
    print("-"*68)
    for r in all_results:
        bv  = r.get('best_val_loss', float('nan'))
        tl  = r['final_test_loss'] if r['final_test_loss'] is not None else float('nan')
        ppl = np.exp(tl) if not np.isnan(tl) else float('nan')
        print(f"{r['model_name']:<22} {r['hidden_size']:>6} {bv:>10.4f} {tl:>10.4f} {ppl:>11.2f}")
    print("="*68)


# ── Plot the baseline on its own (LSTM results added later) ───────────────────
plot_comparison([results], save_path=book_dir / "model_comparison.png")