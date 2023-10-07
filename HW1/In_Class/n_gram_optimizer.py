import torch
import random

class NGramModel:
    def __init__(self, vocab: set):
        self.vocab = vocab
        self.vocab_token_map = {self.vocab[i]: [0 for _ in range(len(self.vocab))] for i in range(len(self.vocab))}
        self.params = torch.random(len(self.vocab))
        self.optimizer = torch.optim.SGD(self.params, lr=0.001, momentum=0.9)

    def forward(self):
        y_pred = torch.nn.Softmax(self.params)
        return y_pred

    def loss_fn(self, y_pred, target):
        residual = torch.nn.CrossEntropyLoss(y_pred, target)
        return residual

    def train(self, epochs):
        epoch = 0
        while epoch < epochs:
            epoch += 1
            for token in self.corpus:
                y_pred = self.forward()
                loss = self.loss_fn(y_pred, self.vocab_token_map[token])
                loss.backward()
                self.optimizer.step()



