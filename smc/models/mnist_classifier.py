import torch
import torch.nn as nn
import torch.nn.functional as F

from smc import DEVICE

from .base_model import BaseModel


class MNIST_Net(BaseModel):
    def __init__(self, load=None):
        super(MNIST_Net, self).__init__()
        self.name = "MNIST"

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        self.average_input = nn.Parameter(torch.randn(28, 28))  # type: ignore

        if load is not None:
            self.load_model(name=load)

    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def loss(self, batch):

        input, target = batch  # pylint: disable=redefined-builtin
        output = self.forward(input.to(DEVICE))
        return F.nll_loss(output, target.to(DEVICE))

    def sample_input(self, n=1):

        eps = torch.randn((n, 28, 28))
        return torch.clamp(self.average_input + (1 * eps), min=0)

    def validation(self, batch):

        input, target = batch  # pylint: disable=redefined-builtin
        output = self.forward(input)
        return {"nll": F.nll_loss(output, target)}
