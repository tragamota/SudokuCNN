import numpy as np
import torch

from model import SudokuCNN

if __name__ == "__main__":
    model = SudokuCNN()
    model.load_state_dict(torch.load("sudoku_CNN.pth"))

    input = input("Enter your Sudoku: ")
    sudoku = np.array([int(digit) for digit in input]).reshape(1, 9, 9)

    sudoku = sudoku.astype(float)
    sudoku = sudoku / 9
    sudoku -= .5

    sudoku = torch.Tensor(sudoku).unsqueeze(0)

    solution = model(sudoku)
    solution = torch.argmax(solution, dim=3)

    print(solution)