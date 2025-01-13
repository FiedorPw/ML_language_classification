import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ploted_interwal = 20

def plot_streamgraph():
    latver_probs, symk_probs, dwak_probs, letters = calculate_probabilities()

    latver_probs = latver_probs[:ploted_interwal]
    symk_probs = symk_probs[:ploted_interwal]
    dwak_probs = dwak_probs[:ploted_interwal]
    letters = letters[:ploted_interwal]


    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 6))

    # Create x-axis points
    x = np.arange(len(letters))

    # Create y values for stacking
    y = np.vstack([latver_probs, symk_probs, dwak_probs])

    # Create streamgraph
    # baseline='wiggle' creates the streamgraph effect
    ax.stackplot(x, y, labels=['Latver', 'Symk', 'Dwak'], baseline='zero', alpha=0.7)

    # Customize the plot
    ax.set_title('Prawdopodobieństwo że litera jest z danego języka')
    ax.set_xlabel('Litery')
    ax.set_ylabel('Prawdopodobieństwo')

    # Set x-axis ticks to show letters
    ax.set_xticks(x)
    ax.set_xticklabels(letters, rotation=45)

    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# Wczytanie danych z pliku do pythonowego array (pomijając indeks)
def read_data_to_array(filename):
    array = []
    with open(filename, "r") as file:  # Podmień "plik.txt" na ścieżkę do swojego pliku
        for line in file:
            parts = line.strip().split("\t")  # Podział na kolumny według tabulatora
            letter = parts[1].strip('"')  # Usuń zbędne cudzysłowy
            array.append(letter)  # Dodaj literę do tablicy
    return array

def read_arrays():
    plik_latver = "data/dlatver 33.txt"
    plik_symk = "data/dsymk 31.txt"
    plik_dwak = "data/dwak 34.txt"
    hidden_message = "data/message 33.txt"
    latver = read_data_to_array(plik_latver)
    symk = read_data_to_array(plik_symk)
    dwak = read_data_to_array(plik_dwak)
    hidden_message = read_data_to_array(hidden_message)
    return latver, symk, dwak, hidden_message


def print_data_characteristics():
    print(f"{latver_counts}\n{symk_counts}\n{dwak_counts}\n{hidden_message_counts}")
    print("latver sum : ", sum(latver_counts), "\nsymk sum : ", sum(symk_counts), "\ndwak sum : ", sum(dwak_counts), "\nhidden_message sum : ", sum(hidden_message_counts))

def calculate_letter_probabilities(language_counts):
    probabilities_dict = {}
    sum_counts = sum(language_counts)
    for letter in language_counts.index:
        probabilities_dict[letter] = language_counts[letter]/sum_counts
    return probabilities_dict



def calculate_probabilities():
    # liczenie procenta wystąpień każdego języka
    posterior_latver = len(latver_list) / (12000)
    posterior_symk = len(symk_list) / (12000)
    posterior_dwak = len(dwak_list) / (12000)
    posterior_sum = posterior_latver + posterior_symk + posterior_dwak

    latver_letter_prob = calculate_letter_probabilities(latver_counts)
    symk_letter_prob = calculate_letter_probabilities(symk_counts)
    dwak_letter_prob = calculate_letter_probabilities(dwak_counts)

    # dla wizualizacji zapisywanie procentów w czasie
    latver_probs = []
    symk_probs = []
    dwak_probs = []

    # bez sprawdzenia czy litera jest w jakimś języku bo wszyskie majązakres A-F
    for letter in hidden_message:
       # update posterior(mnożmy przez prawdopodobieństwo warunkowe dla każdego przypadku w jakim możę być ten język)
       posterior_latver *= latver_letter_prob[letter]
       posterior_symk *= symk_letter_prob[letter]
       posterior_dwak *= dwak_letter_prob[letter]
       posterior_sum = posterior_latver + posterior_symk + posterior_dwak

       latver_probs.append(posterior_latver / posterior_sum)
       symk_probs.append(posterior_symk / posterior_sum)
       dwak_probs.append(posterior_dwak / posterior_sum)

       print(f"latver: {posterior_latver / posterior_sum:.2%}, "f"symk: {posterior_symk / posterior_sum:.2%}, "f"dwak: {posterior_dwak / posterior_sum:.2%} - {letter} ")

    return latver_probs, symk_probs, dwak_probs, hidden_message


if __name__ == '__main__':
    latver, symk, dwak, hidden_message = read_arrays()
    #usunięcie 'x' z początku każdej listy
    latver_list, symk_list, dwak_list, hidden_message = latver[1:], symk[1:], dwak[1:], hidden_message[1:]

    latver_counts = pd.Series(latver_list).value_counts()
    symk_counts   = pd.Series(symk_list).value_counts()
    dwak_counts   = pd.Series(dwak_list).value_counts()
    hidden_message_counts = pd.Series(hidden_message).value_counts()

    calculate_probabilities()
    plot_streamgraph()
