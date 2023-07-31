import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def preprocess_sudoku(file):
    dataset_content = pd.read_csv(file).iloc[:2000000]

    puzzles = dataset_content['puzzle']
    solutions = dataset_content['solution']

    processed_puzzles = []
    processed_solutions = []

    for puzzle in tqdm(puzzles):
        x = np.array([int(digit) for digit in puzzle]).reshape(1, 9, 9)

        x = x.astype(float)
        x = x / 9
        x -= .5

        processed_puzzles.append(x)

    for solution in tqdm(solutions):
        processed_solutions.append(np.array([hot_encode_digit(int(digit)) for digit in solution]).reshape(9, 9, 10))

    return train_test_split(processed_puzzles, processed_solutions, test_size=0.2, shuffle=True)


def hot_encode_digit(digit):
    encoding = np.zeros(10, dtype=np.float32)
    encoding[digit] = 1

    return encoding