import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

#Models

class CharLSTM(nn.Module): #same as RNN but with cell state added to the hidden state
    def __init__(self, vocab_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden

#Training batches

def text_to_indices(text, char_to_ind): #transforms the text into numbers
    return np.array([char_to_ind[ch] for ch in text], dtype=np.int64)


def get_random_batch(data_indices, seq_length, batch_size, vocab_size): #creating random batches
    max_start = len(data_indices) - seq_length - 1
    starts = np.random.randint(0, max_start, size=batch_size)

    x_batch = []
    y_batch = []

    for start in starts:
        x = data_indices[start:start + seq_length]
        y = data_indices[start + 1:start + seq_length + 1]
        x_batch.append(x)
        y_batch.append(y)

    x_batch = torch.tensor(np.array(x_batch), dtype=torch.long)
    y_batch = torch.tensor(np.array(y_batch), dtype=torch.long) #convert into Pytorch tensors

    x_onehot = torch.nn.functional.one_hot(x_batch, num_classes=vocab_size).float()

    return x_onehot, y_batch

#Train the model

def train_model(model, train_indices, val_indices, vocab_size, seq_length=100,
                batch_size=64, learning_rate=1e-3, num_steps=5000,
                eval_every=500, device="cpu"):

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    for step in range(1, num_steps + 1):
        model.train()

        x_batch, y_batch = get_random_batch(
            train_indices, seq_length, batch_size, vocab_size
        )

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits, _ = model(x_batch) #predicts the scores for every next character

        loss = criterion(
            logits.reshape(-1, vocab_size),
            y_batch.reshape(-1)
        )

        optimizer.zero_grad() #put the gradients back to zero
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5) #avoiding the gradients to explode
        optimizer.step() #modifies the weight of the model 

        if step % eval_every == 0: #we evaluate on the validation set every 500 steps
            val_loss = evaluate_model(
                model, val_indices, vocab_size, seq_length, batch_size, device
            )

            train_losses.append(loss.item())
            val_losses.append(val_loss)

            print(f"step {step}, train loss={loss.item():.4f}, val loss={val_loss:.4f}")

    return train_losses, val_losses

#Evaluate model

def evaluate_model(model, data_indices, vocab_size, seq_length=100,
                   batch_size=64, device="cpu", num_batches=20):

    model.eval() #we're not training the model, we're evaluating it
    criterion = nn.CrossEntropyLoss()
    losses = []

    with torch.no_grad(): #we don't want to change the model by calculating the gradients here
        for _ in range(num_batches):
            x_batch, y_batch = get_random_batch(
                data_indices, seq_length, batch_size, vocab_size
            )

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits, _ = model(x_batch)

            loss = criterion(
                logits.reshape(-1, vocab_size),
                y_batch.reshape(-1)
            )

            losses.append(loss.item())

    return np.mean(losses)


current_file = Path(__file__).resolve()
book_dir     = current_file.parent / "data"
book_fname   = book_dir / "divineComedy.txt"

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


#Experiments

train_indices = text_to_indices(train_data, char_to_ind)
val_indices = text_to_indices(val_data, char_to_ind)
test_indices = text_to_indices(test_data, char_to_ind)

#Hyperparameters

vocab_size = len(unique_chars)
hidden_size = 100
seq_length = 100
batch_size = 64
learning_rate = 1e-3
num_steps = 5000
eval_every = 500

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

#LSTM 2-layer 

lstm2_model = CharLSTM(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    num_layers=2,
    dropout=0.2
)

lstm2_train_losses, lstm2_val_losses = train_model(
    lstm2_model,
    train_indices,
    val_indices,
    vocab_size,
    seq_length=seq_length,
    batch_size=batch_size,
    learning_rate=learning_rate,
    num_steps=num_steps,
    eval_every=eval_every,
    device=device
)

lstm2_test_loss = evaluate_model(
    lstm2_model,
    test_indices,
    vocab_size,
    seq_length=seq_length,
    batch_size=batch_size,
    device=device
)

def generate_text(model, start_char, char_to_ind, ind_to_char, vocab_size,
                  length=500, temperature=1.0, device="cpu"):
    model.eval()

    current_index = char_to_ind[start_char]
    generated = [start_char]
    hidden = None

    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor([[current_index]], dtype=torch.long).to(device)
            x_onehot = torch.nn.functional.one_hot(
                x, num_classes=vocab_size
            ).float()

            logits, hidden = model(x_onehot, hidden)

            logits = logits[0, -1, :] / temperature
            probabilities = torch.softmax(logits, dim=0)

            next_index = torch.multinomial(probabilities, num_samples=1).item()
            next_char = ind_to_char[next_index]

            generated.append(next_char)
            current_index = next_index

    return "".join(generated)

# ── Hidden-size experiment ────────────────────────────────────────────────────
 
HIDDEN_SIZES = [60, 70, 80, 90, 110, 120, 130, 140, 150]
 
all_results       = []   # final summary rows
all_train_curves  = {}   # step → train loss  (for plotting)
all_val_curves    = {}   # step → val loss    (for plotting)
 
for hs in HIDDEN_SIZES:
    print(f"\n{'='*60}")
    print(f"  Training 2-layer LSTM  |  hidden_size = {hs}")
    print(f"{'='*60}")
 
    model = CharLSTM(
        vocab_size=vocab_size,
        hidden_size=hs,
        num_layers=2,
        dropout=0.2
    )
 
    # Count trainable parameters so we can report model size
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")
 
    train_losses, val_losses = train_model(
        model, train_indices, val_indices, vocab_size,
        seq_length=seq_length, batch_size=batch_size,
        learning_rate=learning_rate, num_steps=num_steps,
        eval_every=eval_every, device=device
    )
 
    test_loss        = evaluate_model(model, test_indices, vocab_size,
                                      seq_length=seq_length, batch_size=batch_size,
                                      device=device)
    test_perplexity  = np.exp(test_loss)
 
    print(f"\n  hidden={hs}  test_loss={test_loss:.4f}  test_perplexity={test_perplexity:.2f}")
 
    # ── Generate a short sample ───────────────────────────────────────────────
    sample = generate_text(model, "T", char_to_ind, ind_to_char, vocab_size,
                            length=200, temperature=0.8, device=device)
    print(f"\n  Sample text:\n{sample}\n")
 
    all_results.append({
        "hidden_size":      hs,
        "num_params":       num_params,
        "test_loss":        round(test_loss, 4),
        "test_perplexity":  round(float(test_perplexity), 4),
        "train_losses":     [round(v, 4) for v in train_losses],
        "val_losses":       [round(v, 4) for v in val_losses],
    })
 
    all_train_curves[hs] = train_losses
    all_val_curves[hs]   = val_losses
 
# ── Print final comparison table ──────────────────────────────────────────────
 
steps_logged = list(range(eval_every, num_steps + 1, eval_every))   # [500, 1000, …, 5000]
 
print("\n" + "="*70)
print("FINAL RESULTS — hidden-size experiment (2-layer LSTM, 5 000 steps)")
print("="*70)
print(f"{'Hidden':>8}  {'Params':>10}  {'Test Loss':>10}  {'Perplexity':>12}")
print("-"*50)
for r in all_results:
    print(f"{r['hidden_size']:>8}  {r['num_params']:>10,}  {r['test_loss']:>10.4f}  {r['test_perplexity']:>12.2f}")
 
# ── Save results to JSON (for the visualisation dashboard) ────────────────────
output = {
    "steps":   steps_logged,
    "results": all_results,
}
json_path = Path(__file__).parent / "hidden_size_results.json"
with open(json_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {json_path}")