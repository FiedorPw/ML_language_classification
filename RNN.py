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
accelerator = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(accelerator)



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
        return torch.zeros(1,self.hidden_size, device=accelerator)


def one_pass_test(letter):
    input_tensor = letter_to_tensor(letter)
    hidden_tensor = rnn.init_hidden()
    output, next_hidden = rnn(input_tensor, hidden_tensor)
    print("rozmiar outputu - ilość klas",output.size())
    print("rozmiar wektora ukrytego", next_hidden.size())
    return output

def sequence_pass_test(rnn, input_tensor):
    print("rozmiar inputu - batch size, 1, ilość liter", input_tensor.size())
    hidden_tensor = rnn.init_hidden()
    for i, letter in enumerate(letters):
        output, hidden_tensor = rnn(input_tensor[i], hidden_tensor)
    print("rozmiar outputu - ilość klas", output.size())
    return output

def category_from_output(output):
    # print("output procentowy", output)
    category_idx = torch.argmax(output).item()
    # print("indeks klasy", category_idx)
    return kategorie[category_idx]

def train(rnn, criterion, optimizer, category_tensor, sequence_tensor):
    hidden = rnn.init_hidden()
    sequence_tensor = sequence_tensor.to(accelerator)
    category_tensor = category_tensor.to(accelerator)

    output = None
    for i in range(sequence_tensor.size(0)):
        output, hidden = rnn(sequence_tensor[i], hidden)
        # uczenie
    loss = criterion(output, category_tensor) # loss function
    optimizer.zero_grad() # zerowanie gradientów
    loss.backward() # liczenie gradientów
    optimizer.step() # aktualizacja wa

    return output, loss.item() # loss jako floatkk


def training_loop(rnn, criterion, optimizer, n_iters, sequence_length):
    current_loss = 0
    all_losses = []
    all_tests = []
    plot_steps, test_steps = 500, 500

    for i in range(n_iters):
        category, sequence , category_tensor, sequence_tensor = random_training_sequence(sequence_length=sequence_length)

        output, loss = train(rnn, criterion, optimizer, category_tensor, sequence_tensor)
        current_loss += loss

        if i % plot_steps == 0:
            all_losses.append(current_loss / plot_steps) # średnia z ostatnich plot_steps kroków
            current_loss = 0

        if i % test_steps == 0:

            guess = category_from_output(output)
            correct = f"✓" if guess == category else f"✗ ({category})"
            test_score = test_accuracy(rnn, sequence_length)
            all_tests.append(test_score)
            print(f"{i} {i/n_iters*100:.1f}% ({guess} {correct}) {loss:.4f}\n")
                        # procent treningu

    plt.figure()
    plt.plot(all_losses)
    plt.plot(all_tests)
    plt.legend()
    plt.show()

def predict(rnn, sequence_tensor, sequence_length):

    with torch.no_grad():
        hidden = rnn.init_hidden()
        output = torch.zeros(1, N_CATEGORIES)
        for i in range(sequence_tensor.size(0)):
            output, hidden = rnn(sequence_tensor[i], hidden)

        guess = category_from_output(output)
        # print(guess)
        return guess

def test_accuracy(rnn, sequence_length):
    correct = 0
    n_test = 100
    for i in range(n_test):
        category, sequence, category_tensor, sequence_tensor = random_training_sequence(sequence_length=sequence_length, dataset="test")
        output = predict(rnn, sequence_tensor, sequence_length)
        if output == category:
            correct += 1
    accuracy = correct/n_test
    print(f"Accuracy: {accuracy * 100}%")
    return accuracy

def main():
    # 1) Create model
    rnn = RNN(N_LETTERS, N_HIDDEN, N_CATEGORIES)
    rnn.to(accelerator)
    # 2) Define loss function
    criterion = nn.NLLLoss()
    # 3) Define optimizer
    learning_rate = 0.0002
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    # 4) Run training
    training_loop(rnn, criterion, optimizer, n_iters=30_000, sequence_length=22)

if __name__ == '__main__':
    # latver, symk, dwak, hidden_message = read_arrays()
    # # inicjalizacja sieci
    # # one_pass_test('A')
    # sequence_output = sequence_bpass_test('ABCD')
    # print(category_from_output(sequence_output))
    #
    main()
