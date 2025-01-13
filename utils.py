# data: https://download.pytorch.org/tutorial/data.zip
import io
import os
import unicodedata
import string
import glob

import torch
import random
from transform_data import load_data

N_LETTERS = 6
ALL_LETTERS = "ABCDEF"

latver, symk, dwak, hidden_message = read_arrays()
#usunięcie 'x' z początku każdej listy
latver_list, symk_list, dwak_list, hidden_message = latver[1:], symk[1:], dwak[1:], hidden_message[1:]

latver_counts = pd.Series(latver_list).value_counts()
symk_counts   = pd.Series(symk_list).value_counts()
dwak_counts   = pd.Series(dwak_list).value_counts()
hidden_message_counts = pd.Series(hidden_message).value_counts()

category_lines = {
    'symk': symk_list,
    'latver': latver_list,
    'dwak': dwak_list
}

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )

# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return ALL_LETTERS.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor


def random_training_example(category_lines, all_categories):

    def random_choice(a):
        random_idx = random.randint(0, len(a) - 1)
        return a[random_idx]

    category = random_choice(category_lines.keys())
    line = random_choice(category_lines[category])
    TUTAJ FIX TO NA GÓRZE ZMNIENIONE A NA DOLE JESZCZE NIE (TRZEBA DOSTOSOWAĆ DO NOWEGO ŁADOWANIA DANYCH Z CATEGORY_LINES)
    category_tensor = torch.tensor([list(category_lines.keys()).index(category)], dtype=torch.long)

    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor



if __name__ == '__main__':
    ALL_LETTERS = string.ascii_letters + " .,;'"
    print("N_LETTERS is of type:", type(ALL_LETTERS))

    # print(ALL_LETTERS)
    # print(unicode_to_ascii('Ślusàrski'))

    # category_lines, all_categories = load_data()
    # print(category_lines['Italian'][:5])

    # print(letter_to_tensor('J')) # [1, 57]
    # print(line_to_tensor('Jones').size()) # [5, 1, 57]
