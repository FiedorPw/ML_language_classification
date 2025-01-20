# data: https://download.pytorch.org/tutorial/data.zip
import unicodedata
import pandas as pd
import torch
import random
from transform_data import read_arrays

N_LETTERS = 6







ALL_LETTERS = "ABCDEF"
# random_seqence_length = random.randint(3, 7)

latver, symk, dwak, hidden_message = read_arrays()
#usunięcie 'x' z początku każdej listy
latver_list, symk_list, dwak_list, hidden_message = latver[1:], symk[1:], dwak[1:], hidden_message[1:]

latver_counts = pd.Series(latver_list).value_counts()
symk_counts   = pd.Series(symk_list).value_counts()
dwak_counts   = pd.Series(dwak_list).value_counts()
hidden_message_counts = pd.Series(hidden_message).value_counts()

all_categories = {
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
def line_to_tensor(string_letters):
    tensor = torch.zeros(len(string_letters), 1, N_LETTERS)
    for i, letter in enumerate(string_letters):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor


def random_training_sequence(all_categories = all_categories,
    sequence_length = 5,
    dataset = "train",
    train_test_split = 0.8):


    def random_element_sequence(category, sequence_length):
        category_list_length = len(all_categories[category])
        split_id = int(category_list_length * train_test_split)

        if dataset == "train":
            #przykład z od zera do x% danych treningowych

            random_idx = random.randint(0, split_id)
        elif dataset == "test":
            #przykład z x% danych testowych do końca
            random_idx = random.randint(split_id,category_list_length)
        else:
            print("test/train split error")

        return all_categories[category][random_idx:random_idx + sequence_length]  # element z kategori na pozycji id


    category = random.choice(list(all_categories.keys()))
    index_category = list(all_categories.keys()).index(category)
    category_tensor = torch.tensor([index_category], dtype=torch.long)

    sequence = random_element_sequence(category, sequence_length)
    sequence_tensor = line_to_tensor(sequence)

    #       nazwa    sekwencja_liter(arr) indeks_kategorii  sekwencja_liter(tensor
    return category, sequence, category_tensor, sequence_tensor



if __name__ == '__main__':
    category, sequence, category_tensor, sequence_tensor = random_training_sequence(all_categories, 5, dataset = "test")
    print(category)
    print(sequence)
    print(category_tensor)
    print(sequence_tensor)


    # print(letter_to_tensor('J')) # [1, 57]
    # print(line_to_tensor('Jones').size()) # [5, 1, 57]
