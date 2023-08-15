import torch

from torch.utils.data import Dataset

class SudokuDataset(Dataset):

    def __init__(self, puzzles, solutions):

        self.puzzles = puzzles
        self.solutions = solutions

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        x = self.puzzles[idx]
        y = self.solutions[idx]

        return torch.Tensor(x), torch.tensor(y)
