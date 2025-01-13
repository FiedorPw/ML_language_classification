import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import letter_to_index, letter_to_tensor, line_to_tensor
from transform_data import read_arrays

N_LETTERS = 6
n_categories = 3
n_hidden = 128

kategorie = {
    0: "latveriański",
    1: "symkariański",
    2: "wakandyjski"
}

# all letters, n letters
# load data letter to tensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        # input to hidden
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # input to output
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1) # rozmiar embedingu

    def forward(self, input, hidden):
        combined = torch.cat((input,hidden),1) # along dim 1

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output,hidden

    def init_hidden(self):
        return torch.zeros(1,self.hidden_size)

def one_pass_test(letter):
    input_tensor = letter_to_tensor(letter)
    hidden_tensor = rnn.init_hidden()
    output, next_hidden = rnn(input_tensor, hidden_tensor)
    print("rozmiar outputu - ilość klas",output.size())
    print("rozmiar wektora ukrytego", next_hidden.size())
    return output

def sequence_pass_test(letters):
    input_tensor = line_to_tensor(letters)
    print("rozmiar inputu - batch size, 1, ilość liter", input_tensor.size())
    hidden_tensor = rnn.init_hidden()
    for i, letter in enumerate(letters):
        output, hidden_tensor = rnn(input_tensor[i], hidden_tensor)
    print("rozmiar outputu - ilość klas", output.size())
    return output

def category_from_output(output):
    # print("output procentowy", output)
    category_idx = torch.argmax(output).item()
    print("indeks klasy", category_idx)
    return kategorie[category_idx]

criterion = nn.NLLLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

def train(line_tensor, category_tensor):
    hidden = rnn.init_hidden()
    for i range in range(line_tensor()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
        # uczenie
        loss = criterion(output, category_tensor) # loss function
        optimizer.zero_grad() # zerowanie gradientów
        loss.backward() # liczenie gradientów
        optimizer.step() # aktualizacja wag

    return output, loss.item()

current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iters = 100000
for i in range(n_iters):
    category, line, category_tensor, line_tensor = random_training_example()






if __name__ == '__main__':
    latver, symk, dwak, hidden_message = read_arrays()
    # inicjalizacja sieci
    rnn = RNN(N_LETTERS, n_hidden, n_categories)
    # one_pass_test('A')
    sequence_output = sequence_pass_test('ABCD')
    print(category_from_output(sequence_output))
