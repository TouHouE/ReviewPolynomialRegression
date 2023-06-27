import numpy as np
import torch
import matplotlib.pyplot as plt


def get_sign(num, is_final, ratio):
    if is_final:
        return f'{num:.{ratio}f}' if num > 0 else f'{num:.{ratio}f}'
    return f'+ {num:.{ratio}f}' if num > 0 else f'- {abs(num):.{ratio}f}'

def coef_list2str(coef_list, latex=False, ratio=4):
    """
        Just combine the coefficient list with the x-polynomial as a string.
    :param coef_list: a sequence of coefficients of the polynomial, coef_list[i] meaning c_i * x ^ (i + 1)
    :param latex: Whether the output string follows LaTeX format or not.
    :param ratio: show how many digits a float number has.
    :return:
    """

    results = ''
    max_order = len(coef_list)

    for order, coef in enumerate(coef_list):
        is_final = max_order - 1 == order
        if order > 0:
            x_sign = f'x^{order + 1}' if latex else f' * x ^ {order + 1}'
        else:
            x_sign = f'x' if latex else f' * x'
        with torch.no_grad():
            results = f'{get_sign(coef.item(), is_final, ratio)}{x_sign} {results}'
    return results



def make_poly_data(num_data: int, max_order: int, range_info: dict, ratio: int, bias=True):
    """
        Using the method to produce the sample data for training a polynomial model.
        The Return is a dictionary,
        x -> data, y -> label, coef_list -> data pseudo-model, b -> bias
    :param num_data: number of data, is int
    :param max_order: the maximum of polynomial order, is int
    :param range_info: must contain coefficient range("coef_range"), x range("x_range"), if using bias, using "b_range" as keyword.
        is dict
    :param ratio: how many digits with float number, is int
    :param bias: use bias or not, is boolean.
    :return:
    """
    coef_range = range_info['coef_range']
    x_range = range_info['x_range']

    scale = 10 ** ratio
    coef_list = np.random.randint(coef_range[0] * ratio, coef_range[1] * ratio, (max_order)).astype(np.float32) / scale
    if bias:
        b_range = range_info['b_range']
        b = np.random.randint(b_range[0] * scale, b_range[1] * scale, (num_data, 1)).astype(np.float32) / scale
    else:
        b = np.zeros((num_data, 1)).astype(np.float32)

    x = np.random.randint(x_range[0] * scale, x_range[1] * scale, (num_data, 1)).astype(np.float32) / scale
    y = np.zeros((num_data, 1)).astype(np.float32)

    for order, coef in enumerate(coef_list):
        y += x ** (order + 1) * coef

    return {
        'x': x,
        'y': y,
        'bias': b,
        'coef_list': coef_list
    }


def show_regression(x, y, model, epoch, show=True, **kwargs):
    with torch.no_grad():
        plt.figure(figsize=(10, 8))
        plt.axhline(y=0)
        plt.axvline(x=0)
        plt.scatter(x, y, label='Sample')
        x_for_show = torch.from_numpy(np.arange(-10, 10).astype(np.float32)).cuda(0)
        y_for_show = model(x_for_show)
        plt.plot(x_for_show.cpu(), y_for_show.cpu(), label='Regression Line', color='red')
        plt.title(f'Epoch: {epoch}\n$Func:{model.get_polynomial(True)}$')
        plt.legend()
        if show:
            plt.show()
        else:
            plt.savefig(f'{kwargs["path"]}/status_{epoch}.jpg')
            plt.close()

