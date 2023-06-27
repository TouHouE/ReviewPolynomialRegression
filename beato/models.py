import torch
from torch import nn

from beato.utils import coef_list2str, get_sign


class Polynomial(nn.Module):
    def __init__(self, max_order=1, bias=True):
        super().__init__()
        self.max_order = max_order
        self.coefficient = nn.ParameterList([nn.Parameter(torch.zeros(1), requires_grad=True) for _ in range(max_order)])
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=bias)

        for c in self.coefficient:
            nn.init.normal(c)
        if bias:
            nn.init.normal(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: The input shape is (N) or (N, 1)
        :return:
        """
        x_sum = x * self.coefficient[0]
        for order, coef in enumerate(self.coefficient, 1):
            x_sum += x ** (order + 1) * coef

        return x_sum + self.bias

    def get_polynomial(self, latex=False, ratio=4):
        poly_part = coef_list2str(self.coefficient, latex, ratio)
        return f'{poly_part} {get_sign(self.bias.item(), False, ratio)}'

if __name__ == '__main__':
    import numpy as np
    reg = Polynomial(3, bias=False)
    dummy_x = torch.from_numpy(np.random.randint(0, 10, 10).astype(np.float32) / 10)
    dummy_y = reg(dummy_x)

    out = reg.get_polynomial()
    print(out)