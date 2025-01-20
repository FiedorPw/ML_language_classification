import torch
import torch.nn as nn
from utils import letter_to_index, letter_to_tensor, line_to_tensor, random_training_sequence
import matplotlib.pyplot as plt


from transform_data import read_arrays

N_LETTERS = 6
N_CATEGORIES = 3
N_HIDDEN = 128

kategorie = {
    0: "latveriański",
    1: "symkariański",
    2: "wakandyjski"
}

torch.autograd.set_detect_anomaly(True)
# accelerator = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def train(rnn, criterion, optimizer, category_tensor, sequence_tensor):
    hidden = rnn.init_hidden()
    output = None
    for i in range(sequence_tensor.size(0)):
        output, hidden = rnn(sequence_tensor[i], hidden)
        # uczenie
    loss = criterion(output, category_tensor) # loss function
    optimizer.zero_grad() # zerowanie gradientów
    loss.backward() # liczenie gradientów
    optimizer.step() # aktualizacja wag

    return output, loss.item() # loss jako floatkk


def training_loop(rnn, criterion, optimizer, n_iters=100_000):
    current_loss = 0
    all_losses = []
    plot_steps, print_steps = 1000, 5000
    n_iters = 100_000

    for i in range(n_iters):
        category, sequence , category_tensor, sequence_tensor = random_training_sequence(sequence_length=15)

        output, loss = train(rnn, criterion, optimizer, category_tensor, sequence_tensor)
        current_loss += loss

        if i % plot_steps == 0:
            all_losses.append(current_loss / plot_steps) # średnia z ostatnich plot_steps kroków
            current_loss = 0

        if i % print_steps == 0:
            guess = category_from_output(output)
            correct = f"✓l" if guess == category else f"✗ ({category})"
            print(f"{i} {i/n_iters*100:.1f}% ({guess} {correct}) {loss:.4f}")
                        # procent treningu

    plt.figure()
    plt.plot(all_losses)
    plt.show()


def main():
    # 1) Create model
    rnn = RNN(N_LETTERS, N_HIDDEN, N_CATEGORIES)
    # 2) Define loss function
    criterion = nn.NLLLoss()
    # 3) Define optimizer
    learning_rate = 0.001
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    # 4) Run training
    training_loop(rnn, criterion, optimizer)

if __name__ == '__main__':
    # latver, symk, dwak, hidden_message = read_arrays()
    # # inicjalizacja sieci
    # # one_pass_test('A')
    # sequence_output = sequence_pass_test('ABCD')
    # print(category_from_output(sequence_output))
    #
    main()
