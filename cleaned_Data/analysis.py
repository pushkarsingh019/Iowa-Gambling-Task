import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import unique

data = pd.read_csv("pushkar_data.csv")
all_decks = data['mouse.clicked_name'].unique()
all_deck_proportions = []
number_of_choices = len(data['mouse.clicked_name'])
participant = "pushkar"

# functions
def choice_proportions(categories, choices):
    plt.bar(categories, choices)
    plt.title("Deck Choice Comparison")
    plt.xlabel('Decks')
    plt.ylabel('Number of choices')
    plt.show()

def plot_deck_selection_over_trials(data):
    deck_mapping = {'deck_A': 1, 'deck_B': 2, 'deck_C': 3, 'deck_D': 4}
    # Map the 'mouse.clicked_name' column to these numerical values
    data['deck_num'] = data['mouse.clicked_name'].map(deck_mapping)

    plt.figure(figsize=(10,6))
    plt.scatter(data.index, data['deck_num'], alpha=0.7, label=f'Deck Selection')
    plt.plot(data.index, data['deck_num'], alpha=0.5, label='Change in Choices', linestyle='-', marker='', color='darkblue')
    plt.yticks([1, 2, 3, 4], ['Deck A', 'Deck B', 'Deck C', 'Deck D'])
    plt.xlabel('Trial')
    plt.ylabel('Deck')
    plt.title(f"Deck Selection Over Trials for {participant}")
    plt.grid(axis='x')
    plt.legend()
    plt.show()

def plot_deck_proportions_per_block(data, trials_per_block=20):
    deck_proportions = {'deck_A': [], 'deck_B': [], 'deck_C': [], 'deck_D': []}

    total_trials = len(data)
    total_blocks = total_trials // trials_per_block

    for block in range(total_blocks):
        block_data = data.iloc[block*trials_per_block:(block+1)*trials_per_block]
        for deck in deck_proportions.keys():
            proportion = sum(block_data['mouse.clicked_name'] == deck) / trials_per_block
            deck_proportions[deck].append(proportion)

    plt.figure(figsize=(10,6))

    for deck, proportions in deck_proportions.items():
        line_style = 'dotted' if deck in ['deck_A', 'deck_C'] else 'solid'
        plt.plot(range(1, total_blocks + 1), proportions, label=deck, linestyle=line_style)

    plt.xticks(range(1, total_blocks + 1))
    plt.xlabel('Block')
    plt.ylabel('Proportion of Selections')
    plt.title('Proportions of Deck Selections Per Block')
    plt.legend()
    plt.show()


def learning_curve(data, baseline=2000):
    # Ensure "total" column exists in dataframe
    if 'total' not in data.columns:
        print("The DataFrame must contain a 'total' column.")
        return
    
    # Adjusting "total" values against the baseline
    adjusted_totals = data['total'] - baseline

    # Setting up the figure and axes
    plt.figure(figsize=(10, 6))

    # Plotting adjusted "total" as a line graph
    plt.plot(data.index, adjusted_totals, label=f'Total - {baseline}', color='red', marker='o')

    # Adding titles and labels
    plt.title(f'Adjusted Total Values Over Time (Base {baseline})')
    plt.xlabel('Index')
    plt.ylabel(f'Total - {baseline}')
    plt.axhline(y=0, color='black', linewidth=1)
    plt.legend()
    plt.grid(True)

    plt.show()

# Proportion of deck choices
for deck in all_decks:
    deck_choices_length = len(data[data['mouse.clicked_name'] == deck])
    proportion = deck_choices_length / number_of_choices
    all_deck_proportions.append(proportion)

# usage
choice_proportions(all_decks, all_deck_proportions)
plot_deck_selection_over_trials(data)
plot_deck_proportions_per_block(data)
learning_curve(data)
