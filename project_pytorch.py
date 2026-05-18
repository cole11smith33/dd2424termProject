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

test_loss = evaluate_model(model, test_indices, vocab_size, seq_length, batch_size, device) #calculates the final performance on the test set
test_perplexity = np.exp(test_loss) #another way to express the loss 

