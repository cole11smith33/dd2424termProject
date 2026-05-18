import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import copy

seed = 42
rng  = np.random.default_rng(seed)

#read data, same method as in previous assignments in this course
current_file = Path(__file__).resolve()
book_dir     = current_file.parent 
book_fname   = book_dir / "trainingData.txt"

fid = open(book_fname, "r")
book_data = fid.read()
fid.close()

unique_chars = sorted(list(set(book_data)))
K = len(unique_chars)

char_to_ind = {ch: i  for i, ch in enumerate(unique_chars)}
ind_to_char = {i:  ch for i, ch in enumerate(unique_chars)}

m = 100 # hidden state dimensionality
seq_length = 25 # training sequence length
eta = 1e-3 # Adam learning rate
beta1 = 0.9
beta2 = 0.999
eps   = 1e-8

def init_rnn(m, K, rng):
    RNN = {}
    RNN['b'] = np.zeros((m, 1)) # (m,1)
    RNN['c'] = np.zeros((K, 1)) # (K,1)
    RNN['U'] = (1/np.sqrt(2*K)) * rng.standard_normal((m, K)) # (m,K)
    RNN['W'] = (1/np.sqrt(2*m)) * rng.standard_normal((m, m)) # (m,m)
    RNN['V'] = (1/np.sqrt(m)) * rng.standard_normal((K, m)) # (K,m)
    return RNN

RNN = init_rnn(m, K, rng)

#Helper function
def chars_to_onehot(chars):
    T = len(chars)
    X = np.zeros((K, T))
    for t, ch in enumerate(chars):
        X[char_to_ind[ch], t] = 1.0
    return X

 # generates n characters from the RNN
def synthesize(model, initial_hidden_state, start_input, sequence_length):
    hidden_state = initial_hidden_state.copy()
    current_input = start_input.copy()
    generated_indices = []

    
    vocab_size = model['U'].shape[1]

    for _ in range(sequence_length):
        hidden_pre_activation = model['W'] @ hidden_state + model['U'] @ current_input + model['b']
        hidden_state = np.tanh(hidden_pre_activation)
        
        logits = model['V'] @ hidden_state + model['c']
        logits_normalized = logits - np.max(logits)
        probabilities = np.exp(logits_normalized)
        probabilities /= np.sum(probabilities)

        cumulative_probs = np.cumsum(probabilities)
        random_threshold = rng.uniform()
        sampled_index = int(np.where(cumulative_probs > random_threshold)[0][0])
        generated_indices.append(sampled_index)

        current_input = np.zeros((vocab_size, 1))
        current_input[sampled_index, 0] = 1.0

    return generated_indices


def indices_to_string(indices):
    return "".join(ind_to_char[i] for i in indices)


def forward(model, inputs, targets, initial_hidden_state):
    sequence_length = inputs.shape[1]
    hidden_dim = model['W'].shape[0]
    vocab_size = model['V'].shape[0]
    
    hidden_states = np.zeros((hidden_dim, sequence_length + 1))
    hidden_pre_activations = np.zeros((hidden_dim, sequence_length))
    probabilities = np.zeros((vocab_size, sequence_length))
    
    hidden_states[:, [0]] = initial_hidden_state
    total_loss = 0.0

    for t in range(sequence_length):
        input_t = inputs[:, t:t+1]
        
        hidden_pre_activations[:, t:t+1] = (model['W'] @ hidden_states[:, t:t+1] + model['U'] @ input_t + model['b'])
        
        hidden_states[:, t+1:t+2] = np.tanh(hidden_pre_activations[:, t:t+1])
        
        logits = model['V'] @ hidden_states[:, t+1:t+2] + model['c']
        logits_stable = logits - np.max(logits)
        prob_t = np.exp(logits_stable) / np.sum(np.exp(logits_stable))
        probabilities[:, t:t+1] = prob_t
        
        #cross entropy loss for the current time step
        target_index = np.argmax(targets[:, t])
        total_loss -= np.log(prob_t[target_index, 0])

    average_loss = total_loss / sequence_length
    cache = {
        'inputs': inputs, 
        'targets': targets, 
        'hidden_states': hidden_states, 
        'hidden_pre_activations': hidden_pre_activations, 
        'probabilities': probabilities
    }
    
    return average_loss, cache, hidden_states[:, [sequence_length]]

def backward(model, cache):
    inputs = cache['inputs']
    targets = cache['targets']
    hidden_states = cache['hidden_states']
    probabilities = cache['probabilities']
    sequence_length = inputs.shape[1]

    gradients = {key: np.zeros_like(value) for key, value in model.items()}
    
    grad_hidden_next = np.zeros((model['W'].shape[0], 1))

    for t in reversed(range(sequence_length)):
        
        grad_logits = probabilities[:, t:t+1] - targets[:, t:t+1]
        gradients['V'] += grad_logits @ hidden_states[:, t+1:t+2].T 
        gradients['c'] += grad_logits 

        grad_hidden = model['V'].T @ grad_logits + grad_hidden_next

        grad_pre_activation = grad_hidden * (1 - hidden_states[:, t+1:t+2]**2)

        gradients['W'] += grad_pre_activation @ hidden_states[:, t:t+1].T
        gradients['U'] += grad_pre_activation @ inputs[:, t:t+1].T
        gradients['b'] += grad_pre_activation

        grad_hidden_next = model['W'].T @ grad_pre_activation

    # average the gradients over the sequence length
    for key in gradients:
        gradients[key] /= sequence_length
        gradients[key] = np.clip(gradients[key], -5, 5)

    return gradients

#Altering premade code from torch_gradient_computations_row_wise.py and torch_gradient_computations_column_wise.py

def ComputeGradsWithTorch_col(X, y_indices, h0, RNN):
    
    tau = X.shape[1]
    Xt = torch.from_numpy(X)
    ht = torch.from_numpy(h0)
    
    torch_network = {}
    for kk in RNN.keys():
        torch_network[kk] = torch.tensor(RNN[kk], requires_grad=True)
    apply_tanh = torch.nn.Tanh()
    apply_softmax = torch.nn.Softmax(dim=0)
    
    Hs = torch.empty(h0.shape[0], tau, dtype=torch.float64)
    
    hprev = ht
    for t in range(tau):
        a = torch_network['W'] @ hprev + torch_network['U'] @ Xt[:, t:t+1] + torch_network['b']
        hprev = apply_tanh(a)
        Hs[:, t:t+1] = hprev
 
    Os = torch.matmul(torch_network['V'], Hs) + torch_network['c']   
    P  = apply_softmax(Os)
    
    loss = torch.mean(-torch.log(P[y_indices, np.arange(tau)]))
    loss.backward()
    
    grads = {}
    for kk in RNN.keys():
        grads[kk] = torch_network[kk].grad.numpy()

    return grads


def ComputeGradsWithTorch_row(X, y_indices, h0, RNN):

    X_row = X.T
    h0_row = h0.T
    
    tau = X_row.shape[0]
    Xt = torch.from_numpy(X_row)
    ht = torch.from_numpy(h0_row)
    
    torch_network = {}
    for kk in RNN.keys():
        torch_network[kk] = torch.tensor(RNN[kk], requires_grad=True)

    apply_tanh = torch.nn.Tanh()
    apply_softmax = torch.nn.Softmax(dim=1)
    
    Hs = torch.empty(tau, h0.shape[0], dtype=torch.float64)

    hprev = ht
    for t in range(tau):
        a = hprev @ torch_network['W'].T + Xt[t:t+1, :] @ torch_network['U'].T + torch_network['b'].T
        hprev = apply_tanh(a)
        Hs[t:t+1, :] = hprev
 
    Os = torch.matmul(Hs, torch_network['V'].T) + torch_network['c'].T
    P  = apply_softmax(Os)
    
    loss = torch.mean(-torch.log(P[np.arange(tau), y_indices]))
    loss.backward()
    
    grads = {}
    for kk in RNN.keys():
        grads[kk] = torch_network[kk].grad.numpy()

    return grads

print("Gradient check (m=10, seq_length=25)")
m_check  = 10
RNN_chk  = init_rnn(m_check, K, np.random.default_rng(0))
h0_chk   = np.zeros((m_check, 1))
X_chk    = chars_to_onehot(book_data[:seq_length])
Y_onehot = chars_to_onehot(book_data[1:seq_length+1])

y_indices = np.argmax(Y_onehot, axis=0) 

_, cache_chk, _ = forward(RNN_chk, X_chk, Y_onehot, h0_chk)
my_grads = backward(RNN_chk, cache_chk)

# Test both templates
for name, func in [("Template 1 (Col)", ComputeGradsWithTorch_col), 
                   ("Template 2 (Row)", ComputeGradsWithTorch_row)]:
    
    print(f"\nTesting {name}:")
    torch_grads = func(X_chk, y_indices, h0_chk, RNN_chk)
    
    eps_rel = 1e-20
    for key in ['W', 'U', 'V', 'b', 'c']:
        ga  = my_grads[key]
        gt  = torch_grads[key]
        rel = np.abs(ga - gt) / np.maximum(eps_rel, np.abs(ga) + np.abs(gt))
        print(f"  {key}  max relative error: {rel.max()}")


def init_adam(RNN):
    m_adam = {k: np.zeros_like(v) for k, v in RNN.items()}
    v_adam = {k: np.zeros_like(v) for k, v in RNN.items()}
    return m_adam, v_adam

def adam_update(RNN, grads, m_adam, v_adam, t_adam, eta, beta1, beta2, eps):
    for k in RNN:
        m_adam[k] = beta1 * m_adam[k] + (1 - beta1) * grads[k]
        v_adam[k] = beta2 * v_adam[k] + (1 - beta2) * grads[k]**2
        m_hat = m_adam[k] / (1 - beta1**t_adam)
        v_hat = v_adam[k] / (1 - beta2**t_adam)
        RNN[k] -= eta * m_hat / (np.sqrt(v_hat) + eps)

# Main Training loop
def train(RNN, book_data, n_epochs=3, print_every=100, synth_every=1000, synth_len=200): #trains the RNN for n epochs (3 unless specified otherwise for testing purposes)

    m_adam, v_adam = init_adam(RNN)
    t_adam     = 0 # Adam time step counter
    step       = 0 # overall SGD step counter
    smooth_loss = None

    loss_history = []
    best_loss    = float('inf')
    best_RNN     = None

    for epoch in range(1, n_epochs + 1):
        e      = 0
        hprev  = np.zeros((m, 1))

        while e <= len(book_data) - seq_length - 1:
            X_chars = book_data[e : e + seq_length]
            Y_chars = book_data[e + 1 : e + seq_length + 1]
            X       = chars_to_onehot(X_chars)
            Y       = chars_to_onehot(Y_chars)

            # forward and backward passes
            loss, cache, hprev = forward(RNN, X, Y, hprev)
            grads = backward(RNN, cache)

            # update adam
            t_adam += 1
            adam_update(RNN, grads, m_adam, v_adam, t_adam, eta, beta1, beta2, eps)

            # smooth loss for monitoring
            if smooth_loss is None:
                smooth_loss = loss
            else:
                smooth_loss = 0.999 * smooth_loss + 0.001 * loss

            if step == 0 or (step + 1) % print_every == 0:
                print(f"epoch {epoch}  step {step+1:6d}  smooth_loss={smooth_loss:.6f}")
                loss_history.append((step + 1, smooth_loss))

            # track best model (relevant for last step of report)
            if smooth_loss < best_loss:
                best_loss = smooth_loss
                best_RNN  = copy.deepcopy(RNN)

            # synthesise text from the model at regular intervals
            if step == 0 or (step + 1) % synth_every == 0:
                x0_ = X[:, 0:1].copy()
                idx = synthesize(RNN, hprev, x0_, synth_len)
                print(f"\n  [synth at step {step+1}]")
                print(f"\n{indices_to_string(idx)}\n")

            e    += seq_length
            step += 1

        print(f"\n------------------------End of epoch {epoch}\n")

        hprev = np.zeros((m, 1))

    return loss_history, best_RNN

loss_history, best_RNN = train(
    RNN,
    book_data,
    n_epochs=3,
    print_every=100,
    synth_every=1000,
    synth_len=200,
)

#ploting the smooth loss curve
steps_plot  = [s for s, _ in loss_history]
losses_plot = [l for _, l in loss_history]

plt.figure(figsize=(10, 4))
plt.plot(steps_plot, losses_plot)
plt.xlabel("Update step")
plt.ylabel("Smooth loss")
plt.title("Smooth loss during RNN training (3 epochs)")
plt.tight_layout()
plt.savefig("smooth_loss.png", dpi=150)
plt.show()
print("Loss curve saved to smooth_loss.png")

# generate passage from the best model
h0_best = np.zeros((m, 1))
x0_best = np.zeros((K, 1))
x0_best[char_to_ind['.'], 0] = 1.0
best_indices = synthesize(best_RNN, h0_best, x0_best, 1000)
best_text = indices_to_string(best_indices)

print("Text from generated from best model: ")

print("\n" + best_text + "\n")

with open("best_model_sample.txt", "w") as f:
    f.write(best_text)
print("Sample saved to best_model_sample.txt")