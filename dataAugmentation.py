from pathlib import Path
import random
import re

import torch
import torch.nn as nn
import numpy as np
import nltk
from nltk.corpus import wordnet

nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)



class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=2, dropout=0.0):
        super().__init__()
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
        return self.fc(out), hidden


def text_to_indices(text, char_to_ind):
    return np.array([char_to_ind[ch] for ch in text], dtype=np.int64)


def get_random_batch(data_indices, seq_length, batch_size, vocab_size):
    max_start = len(data_indices) - seq_length - 1
    starts = np.random.randint(0, max_start, size=batch_size)
    x_batch = [data_indices[s:s + seq_length]     for s in starts]
    y_batch = [data_indices[s + 1:s + seq_length + 1] for s in starts]
    x = torch.nn.functional.one_hot(
        torch.tensor(np.array(x_batch), dtype=torch.long), num_classes=vocab_size
    ).float()
    y = torch.tensor(np.array(y_batch), dtype=torch.long)
    return x, y


def evaluate_model(model, data_indices, vocab_size, seq_length=100,
                   batch_size=64, device="cpu", num_batches=20):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_random_batch(data_indices, seq_length, batch_size, vocab_size)
            logits, _ = model(x.to(device))
            losses.append(criterion(logits.reshape(-1, vocab_size), y.to(device).reshape(-1)).item())
    return float(np.mean(losses))


def train_model(model, train_indices, val_indices, vocab_size, seq_length=100,
                batch_size=64, learning_rate=1e-3, num_steps=5000,
                eval_every=500, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []

    for step in range(1, num_steps + 1):
        model.train()
        x, y = get_random_batch(train_indices, seq_length, batch_size, vocab_size)
        logits, _ = model(x.to(device))
        loss = criterion(logits.reshape(-1, vocab_size), y.to(device).reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        if step % eval_every == 0:
            val_loss = evaluate_model(model, val_indices, vocab_size, seq_length, batch_size, device)
            train_losses.append(loss.item())
            val_losses.append(val_loss)
            print(f"step {step}  train={loss.item():.4f}  val={val_loss:.4f}")

    return train_losses, val_losses


def generate_text(model, start_char, char_to_ind, ind_to_char, vocab_size,
                  length=300, temperature=1.0, device="cpu"):
    model.eval()
    current = char_to_ind[start_char]
    generated = [start_char]
    hidden = None
    with torch.no_grad():
        for _ in range(length):
            x = torch.nn.functional.one_hot(
                torch.tensor([[current]], dtype=torch.long), num_classes=vocab_size
            ).float().to(device)
            logits, hidden = model(x, hidden)
            probs = torch.softmax(logits[0, -1, :] / temperature, dim=0)
            current = torch.multinomial(probs, 1).item()
            generated.append(ind_to_char[current])
    return "".join(generated)



def augment_char_noise(text, p=0.005):
    chars = list(text)
    result = []
    for ch in chars:
        r = random.random()
        if r < p / 3:
            pass                              # deletion
        elif r < 2 * p / 3:
            result.append(ch)
            result.append(random.choice(chars))  # insertion
        elif r < p:
            result.append(random.choice(chars))  # substitution
        else:
            result.append(ch)
    return "".join(result)


def _get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def augment_synonym(text, aug_p=0.1):
    words = re.findall(r"(\w+|\W+)", text)
    tags = nltk.pos_tag([w for w in words if w.strip() and w.isalpha()])
    tag_map = {w: t for w, t in tags}
    result = []
    for token in words:
        # skip short words, capitalised words (names/places), 
        # and archaic-looking tokens
        if (token.isalpha() 
                and len(token) > 4
                and token[0].islower()
                and token.isascii()
                and random.random() < aug_p):
            pos = _get_wordnet_pos(tag_map.get(token, "NN"))
            synsets = wordnet.synsets(token, pos=pos)
            candidates = [
                l.name().replace("_", " ")
                for s in synsets for l in s.lemmas()
                if l.name().lower() != token.lower()
            ]
            result.append(random.choice(candidates) if candidates else token)
        else:
            result.append(token)
    return "".join(result)


def augment_chunk_shuffle(text, chunk_size=2000):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    random.shuffle(chunks)
    return "".join(chunks)


def apply_augmentation(text, technique):
    if technique == "baseline":
        return text
    if technique == "char_noise":
        return augment_char_noise(text, p=0.02)
    if technique == "synonym":
        return augment_synonym(text, aug_p=0.1)
    if technique == "chunk_shuffle":
        return augment_chunk_shuffle(text, chunk_size=200)
    raise ValueError(f"Unknown technique: {technique}")



current_file = Path(__file__).resolve()
book_dir = current_file.parent / "data"

with open(book_dir / "train.txt", "r", encoding="utf-8", errors="replace") as f:
    train_raw = f.read()
with open(book_dir / "val.txt", "r", encoding="utf-8", errors="replace") as f:
    val_raw = f.read()
with open(book_dir / "test.txt", "r", encoding="utf-8", errors="replace") as f:
    test_raw = f.read()

unique_chars = sorted(set(train_raw + val_raw + test_raw))
vocab_size = len(unique_chars)
char_to_ind = {ch: i for i, ch in enumerate(unique_chars)}
ind_to_char = {i: ch for i, ch in enumerate(unique_chars)}

val_indices  = text_to_indices(val_raw,  char_to_ind)
test_indices = text_to_indices(test_raw, char_to_ind)

print(f"Vocab size: {vocab_size}  |  train chars: {len(train_raw):,}")


hidden_size  = 135
seq_length   = 100
batch_size   = 64
lr           = 1e-3
num_steps    = 5000
eval_every   = 500
device       = "cuda" if torch.cuda.is_available() else "cpu"

techniques = ["baseline", "char_noise", "synonym", "chunk_shuffle"]
techniques = ["char_noise", "synonym", "chunk_shuffle"]

results = {}

for technique in techniques:
    print(f"\n{'='*60}")
    print(f"  Augmentation: {technique}")
    print(f"{'='*60}")

    augmented_train = apply_augmentation(train_raw, technique)
    augmented_train = "".join(ch for ch in augmented_train if ch in char_to_ind)
    train_indices   = text_to_indices(augmented_train, char_to_ind)

    model = CharLSTM(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=2, dropout=0.4)

    train_model(
        model, train_indices, val_indices, vocab_size,
        seq_length=seq_length, batch_size=batch_size,
        learning_rate=lr, num_steps=num_steps,
        eval_every=eval_every, device=device
    )

    test_loss = evaluate_model(model, test_indices, vocab_size,
                               seq_length=seq_length, batch_size=batch_size, device=device)
    test_perplexity = np.exp(test_loss)

    sample = generate_text(model, "T", char_to_ind, ind_to_char, vocab_size,
                           length=300, temperature=0.5, device=device)

    results[technique] = {
        "test_loss": test_loss,
        "test_perplexity": test_perplexity,
        "sample": sample
    }

    print(f"\nTest loss: {test_loss:.4f}  |  Perplexity: {test_perplexity:.4f}")
    print(f"\nSample:\n{sample}")


print(f"\n{'='*60}")
print(f"{'Technique':<16} {'Test Loss':>10} {'Perplexity':>12}")
print(f"{'-'*40}")
for technique, r in results.items():
    print(f"{technique:<16} {r['test_loss']:>10.4f} {r['test_perplexity']:>12.4f}")