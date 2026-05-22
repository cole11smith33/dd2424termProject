from pathlib import Path
import re
import math
from collections import Counter

import torch
import torch.nn as nn
import numpy as np

#Models

class CharRNN(nn.Module): #input characters -> RNN -> hidden states -> linear layer -> next character scores
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        ) #creates the RNN 
        self.fc = nn.Linear(hidden_size, vocab_size) #transforms the hidden state of size hidden_size into scores for each characters of the vocabulary

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden) #out contains the hidden states for all the characters of the sequence 
        logits = self.fc(out) 
        return logits, hidden #hidden contains the last hidden state


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


test_loss = evaluate_model(rnn_model, test_indices, vocab_size, seq_length, batch_size, device) #calculates the final performance on the test set
test_perplexity = np.exp(test_loss) #another way to express the loss 

lstm2_test_perplexity = np.exp(lstm2_test_loss)

print("LSTM 2-layer test loss:", lstm2_test_loss)
print("LSTM 2-layer test perplexity:", lstm2_test_perplexity)

results = [
    ["LSTM", 2, hidden_size, lstm2_test_loss, lstm2_test_perplexity]
]

print("\nFinal Results")
print("Model\tLayers\tHidden Size\tTest Loss\tTest Perplexity")

for row in results:
    print(f"{row[0]}\t{row[1]}\t{row[2]}\t\t{row[3]:.4f}\t\t{row[4]:.4f}")

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

def tokenize_words(text):
    return re.findall(r"[A-Za-z']+", text.lower())


def get_ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def reference_word_percentage(generated_text, reference_text):
    generated_words = tokenize_words(generated_text)
    reference_words = set(tokenize_words(reference_text))

    if len(generated_words) == 0:
        return 0.0

    matching_words = 0

    for word in generated_words:
        if word in reference_words:
            matching_words += 1

    return 100 * matching_words / len(generated_words)


def ngram_overlap_percentage(generated_text, reference_text, n):
    generated_words = tokenize_words(generated_text)
    reference_words = tokenize_words(reference_text)

    generated_ngrams = get_ngrams(generated_words, n)
    reference_ngrams = set(get_ngrams(reference_words, n))

    if len(generated_ngrams) == 0:
        return 0.0

    matching_ngrams = 0

    for ngram in generated_ngrams:
        if ngram in reference_ngrams:
            matching_ngrams += 1

    return 100 * matching_ngrams / len(generated_ngrams)


def bleu_score(generated_text, reference_text, max_n=4):
    generated_words = tokenize_words(generated_text)
    reference_words = tokenize_words(reference_text)

    if len(generated_words) == 0:
        return 0.0

    precisions = []

    for n in range(1, max_n + 1):
        generated_ngrams = Counter(get_ngrams(generated_words, n))
        reference_ngrams = Counter(get_ngrams(reference_words, n))

        if len(generated_ngrams) == 0:
            precisions.append(1e-9)
            continue

        overlap = 0
        total = 0

        for ngram, count in generated_ngrams.items():
            overlap += min(count, reference_ngrams.get(ngram, 0))
            total += count

        precision = overlap / total if total > 0 else 1e-9
        precisions.append(max(precision, 1e-9))

    reference_length = len(reference_words)
    generated_length = len(generated_words)

    if generated_length > reference_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - reference_length / generated_length)

    log_precision_sum = 0.0

    for precision in precisions:
        log_precision_sum += math.log(precision)

    bleu = brevity_penalty * math.exp(log_precision_sum / max_n)

    return bleu


def evaluate_generated_text(generated_text, reference_text):
    word_percentage = reference_word_percentage(generated_text, reference_text)
    bigram_overlap = ngram_overlap_percentage(generated_text, reference_text, 2)
    trigram_overlap = ngram_overlap_percentage(generated_text, reference_text, 3)
    bleu = bleu_score(generated_text, reference_text)

    return {
        "reference_word_percentage": word_percentage,
        "bigram_overlap": bigram_overlap,
        "trigram_overlap": trigram_overlap,
        "bleu": bleu
    }

print("LSTM 2-layer sample:")
print(generate_text(lstm2_model, "T", char_to_ind, ind_to_char, vocab_size, device=device))

generated_text = generate_text(
    lstm2_model,
    "T",
    char_to_ind,
    ind_to_char,
    vocab_size,
    length=2000,
    temperature=1.0,
    device=device
)

print("\nLong generated sample from LSTM 2-layer:")
print(generated_text)

text_metrics = evaluate_generated_text(generated_text, train_data)

print("\nGenerated Text Evaluation Metrics")
print("Metric\tValue")
print(f"Reference word percentage\t{text_metrics['reference_word_percentage']:.2f}%")
print(f"Bigram overlap\t{text_metrics['bigram_overlap']:.2f}%")
print(f"Trigram overlap\t{text_metrics['trigram_overlap']:.2f}%")
print(f"BLEU score\t{text_metrics['bleu']:.4f}")
print(f"Test perplexity\t{lstm2_test_perplexity:.4f}")


