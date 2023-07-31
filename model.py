import torch
import torch.nn as nn


class SudokuCNN(nn.Module):
    def __init__(self):
        super(SudokuCNN, self).__init__()

        self.layers = nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.Conv2d(1, 512, kernel_size=3),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(512, 512, kernel_size=3),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(512, 512, kernel_size=3),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(512, 512, kernel_size=3),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(512, 512, kernel_size=3),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(512, 512, kernel_size=3),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(512, 512, kernel_size=3),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(512, 512, kernel_size=3),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(512, 512, kernel_size=3),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(512, 512, kernel_size=3),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(512, 512, kernel_size=3),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(512, 512, kernel_size=3),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(512, 512, kernel_size=3),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(512, 512, kernel_size=3),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(512, 512, kernel_size=3),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(512, 512, kernel_size=3),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True),
                                    nn.Conv2d(512, 10, kernel_size=1),
                                    nn.BatchNorm2d(10),
                                    nn.ReLU(True))

        self.fc1 = nn.Linear(512, 81 * 10)

    def forward(self, x):
        x = self.layers(x)

        # x = x.view(-1, 512)
        # x = self.fc1(x).view(-1, 9, 9, 10)

        return x.view(-1, 9, 9, 10)
