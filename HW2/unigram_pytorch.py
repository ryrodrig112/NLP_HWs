"""Pytorch."""
import nltk
import numpy as np
from numpy.typing import NDArray
import torch
from typing import List, Optional
from torch import nn
import matplotlib.pyplot as plt


FloatArray = NDArray[np.float64]


def onehot(
    vocabulary: List[Optional[str]], token: Optional[str]
) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary), 1))
    try:
        idx = vocabulary.index(token)
    except ValueError:
        idx = len(vocabulary) - 1
    embedding[idx, 0] = 1
    return embedding


def logit(x: FloatArray) -> FloatArray:
    """Compute logit (inverse sigmoid)."""
    return np.log(x) - np.log(1 - x)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize vector so that it sums to 1."""
    return x / torch.sum(x)


def loss_fn(p: float) -> float:
    """Compute loss to maximize probability."""
    return -p


class Unigram(nn.Module):
    def __init__(self, V: int):
        super().__init__()

        # construct initial s - corresponds to uniform p
        s0 = logit(np.ones((V, 1)) / V)
        self.s = nn.Parameter(torch.tensor(s0.astype("float32")))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # convert s to proper distribution p
        p = normalize(torch.sigmoid(self.s))

        # compute log probability of input
        return torch.sum(input, 1, keepdim=True).T @ torch.log(p)


def get_true_probabilities():
    vocabulary = [chr(i + ord("a")) for i in range(26)] + [" "]
    text = nltk.corpus.gutenberg.raw("austen-sense.txt").lower()
    corpus_tokens = [char for char in text]
    count_dict = {vocab_token: 0 for vocab_token in vocabulary}

    other_counter = 0
    for token in corpus_tokens:
        if token in count_dict:
            count_dict[token] = count_dict[token] + 1
        else:
            other_counter += 1
    count_dict['N/A'] = other_counter

    n = sum(count_dict.values())

    prob_dict = {key: count_dict[key] / n for key in count_dict.keys()}
    return prob_dict


def plot_wpred_wtrue(w_pred, w_true):
    """Plots predicted and true unigram probabilities for comparison"""
    fig, ax = plt.subplots()
    x = [chr(i + ord("a")) for i in range(26)] + [" ", "N/A"]
    plt.scatter(x, w_pred, label="Predicted Probability (GD)")
    plt.scatter(x, w_true, label="Actual Probability")
    plt.legend()
    plt.xlabel("Token")
    plt.ylabel("Probability")
    plt.title("Predicted and Actual Token Probabilities")
    return fig, ax


def plot_loss_over_time(optimal_loss, losses, num_iters):
    fig, ax = plt.subplots()
    plt.plot([i for i in range(num_iters)], losses, label="Training Loss")
    plt.axhline(optimal_loss, color="grey", linestyle="dotted", label="Optimal Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Loss Over Training Iterations")
    plt.legend()

    return fig, ax

def calculate_optimal_loss(encodings, true_probs):
    opt_loss = -1 * (np.sum(encodings, 1, keepdims=True).T @ np.log(true_probs))
    return opt_loss

def gradient_descent_example():
    """Demonstrate gradient descent."""
    # generate vocabulary
    vocabulary = [chr(i + ord("a")) for i in range(26)] + [" ", None]

    # generate training document
    text = nltk.corpus.gutenberg.raw("austen-sense.txt").lower()

    # tokenize - split the document into a list of little strings
    tokens = [char for char in text]

    # generate one-hot encodings - a V-by-T array
    encodings = np.hstack([onehot(vocabulary, token) for token in tokens])
    # convert training data to PyTorch tensor
    x = torch.tensor(encodings.astype("float32"))

    # define model
    model = Unigram(len(vocabulary))

    # set number of iterations and learning rate
    num_iterations = 20
    learning_rate = 0.1

    # train model
    losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(num_iterations):
        p_pred = model(x)
        loss = -p_pred
        losses.append(float(loss.detach()))
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

    # display results

    print(f"Iterations: {num_iterations}")
    print(f"Learning Rate: {learning_rate}")

    model_probs = model.s
    w_pred = [float(np.exp(list(model_probs.detach().numpy())[i])) for i in range(len(model_probs))]
    w_true = list(get_true_probabilities().values())
    print(w_true)
    opt_loss = calculate_optimal_loss(encodings,w_true)
    fig1, ax1 = plot_wpred_wtrue(w_pred, w_true)
    plt.show()
    plot_loss_over_time(opt_loss, losses, num_iterations)
    plt.show()


    # return model.s


if __name__ == "__main__":
    gradient_descent_example()