from pathlib import Path

import torch
import torch.nn as nn
import numpy as np


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=2, dropout=0.0):
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


def text_to_indices(text, char_to_ind):
    return np.array([char_to_ind[ch] for ch in text], dtype=np.int64)


def get_random_batch(data_indices, seq_length, batch_size, vocab_size):
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
    y_batch = torch.tensor(np.array(y_batch), dtype=torch.long)

    x_onehot = torch.nn.functional.one_hot(
        x_batch,
        num_classes=vocab_size
    ).float()

    return x_onehot, y_batch


def evaluate_model(model, data_indices, vocab_size, seq_length=100,
                   batch_size=64, device="cpu", num_batches=20):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []

    with torch.no_grad():
        for _ in range(num_batches):
            x_batch, y_batch = get_random_batch(
                data_indices,
                seq_length,
                batch_size,
                vocab_size
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


def train_model(model, train_indices, val_indices, vocab_size, seq_length=100,
                batch_size=64, learning_rate=1e-3, num_steps=1000,
                eval_every=250, device="cpu"):

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    for step in range(1, num_steps + 1):
        model.train()

        x_batch, y_batch = get_random_batch(
            train_indices,
            seq_length,
            batch_size,
            vocab_size
        )

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits, _ = model(x_batch)

        loss = criterion(
            logits.reshape(-1, vocab_size),
            y_batch.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        if step % eval_every == 0:
            val_loss = evaluate_model(
                model,
                val_indices,
                vocab_size,
                seq_length=seq_length,
                batch_size=batch_size,
                device=device
            )

            train_losses.append(loss.item())
            val_losses.append(val_loss)

            print(
                f"step {step}, "
                f"train loss={loss.item():.4f}, "
                f"val loss={val_loss:.4f}"
            )

    return train_losses, val_losses


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
                x,
                num_classes=vocab_size
            ).float()

            logits, hidden = model(x_onehot, hidden)

            logits = logits[0, -1, :] / temperature
            probabilities = torch.softmax(logits, dim=0)

            next_index = torch.multinomial(
                probabilities,
                num_samples=1
            ).item()

            next_char = ind_to_char[next_index]

            generated.append(next_char)
            current_index = next_index

    return "".join(generated)


current_file = Path(__file__).resolve()
book_dir = current_file.parent / "data"
book_fname = book_dir / "train.txt"

with open(book_fname, "r", encoding="utf-8", errors="replace") as fid:
    book_data = fid.read()

unique_chars = sorted(list(set(book_data)))
vocab_size = len(unique_chars)

char_to_ind = {ch: i for i, ch in enumerate(unique_chars)}
ind_to_char = {i: ch for i, ch in enumerate(unique_chars)}

n = len(book_data)
train_end = int(0.80 * n)
val_end = int(0.90 * n)

train_data = book_data[:train_end]
val_data = book_data[train_end:val_end]
test_data = book_data[val_end:]

print(f"Vocabulary size : {vocab_size}")
print(f"Total chars     : {n:,}")
print(f"Train           : {len(train_data):,}  ({len(train_data) / n:.1%})")
print(f"Validation      : {len(val_data):,}   ({len(val_data) / n:.1%})")
print(f"Test            : {len(test_data):,}   ({len(test_data) / n:.1%})")

train_indices = text_to_indices(train_data, char_to_ind)
val_indices = text_to_indices(val_data, char_to_ind)
test_indices = text_to_indices(test_data, char_to_ind)


hidden_size = 100
seq_length = 100

grid_num_steps = 5000
grid_eval_every = 500

final_num_steps = 5000
final_eval_every = 500

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


batch_sizes = [32, 64]
learning_rates = [1e-3, 5e-4]
dropouts = [0.0, 0.2, 0.4]

grid_results = []

for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        for dropout in dropouts:
            print("\n----------------------------------------")
            print(
                f"Training 2-layer LSTM with "
                f"batch_size={batch_size}, "
                f"learning_rate={learning_rate}, "
                f"dropout={dropout}"
            )

            model = CharLSTM(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_layers=2,
                dropout=dropout
            )

            train_losses, val_losses = train_model(
                model,
                train_indices,
                val_indices,
                vocab_size,
                seq_length=seq_length,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_steps=grid_num_steps,
                eval_every=grid_eval_every,
                device=device
            )

            final_train_loss = train_losses[-1]
            final_val_loss = val_losses[-1]
            final_val_perplexity = np.exp(final_val_loss)

            grid_results.append({
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "dropout": dropout,
                "train_loss": final_train_loss,
                "val_loss": final_val_loss,
                "val_perplexity": final_val_perplexity
            })

            print(f"Final train loss: {final_train_loss:.4f}")
            print(f"Final val loss: {final_val_loss:.4f}")
            print(f"Final val perplexity: {final_val_perplexity:.4f}")


print("\nGrid Search Results")
print("Batch Size\tLearning Rate\tDropout\tTrain Loss\tVal Loss\tVal Perplexity")

for result in grid_results:
    print(
        f"{result['batch_size']}\t\t"
        f"{result['learning_rate']}\t\t"
        f"{result['dropout']}\t"
        f"{result['train_loss']:.4f}\t\t"
        f"{result['val_loss']:.4f}\t\t"
        f"{result['val_perplexity']:.4f}"
    )

best_result = min(grid_results, key=lambda x: x["val_loss"])

print("\nBest hyperparameters based on validation loss:")
print(f"Batch size: {best_result['batch_size']}")
print(f"Learning rate: {best_result['learning_rate']}")
print(f"Dropout: {best_result['dropout']}")
print(f"Validation loss: {best_result['val_loss']:.4f}")
print(f"Validation perplexity: {best_result['val_perplexity']:.4f}")


print("\n----------------------------------------")
print("Retraining best 2-layer LSTM configuration")

best_model = CharLSTM(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    num_layers=2,
    dropout=best_result["dropout"]
)

best_train_losses, best_val_losses = train_model(
    best_model,
    train_indices,
    val_indices,
    vocab_size,
    seq_length=seq_length,
    batch_size=best_result["batch_size"],
    learning_rate=best_result["learning_rate"],
    num_steps=final_num_steps,
    eval_every=final_eval_every,
    device=device
)

best_test_loss = evaluate_model(
    best_model,
    test_indices,
    vocab_size,
    seq_length=seq_length,
    batch_size=best_result["batch_size"],
    device=device
)

best_test_perplexity = np.exp(best_test_loss)

print("\nFinal Tuned 2-layer LSTM Results")
print(f"Test loss: {best_test_loss:.4f}")
print(f"Test perplexity: {best_test_perplexity:.4f}")

print("\nGenerated sample from tuned 2-layer LSTM:")
print(
    generate_text(
        best_model,
        "T",
        char_to_ind,
        ind_to_char,
        vocab_size,
        length=500,
        temperature=1.0,
        device=device
    )
)
