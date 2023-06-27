import torch
from torch.utils import data as TD
from beato.utils import make_poly_data, show_regression
from beato.models import Polynomial


model = Polynomial(5)
sample = make_poly_data(num_data=2000, max_order=5, ratio=4, range_info={
    'coef_range': [-100, 100],
    'x_range': [-10, 10],
    'b_range': [-5, 5]
})


x, y = torch.from_numpy(sample['x']), torch.from_numpy(sample['y'])
loader = TD.DataLoader(TD.TensorDataset(x, y), batch_size=50, shuffle=True)

model.cuda(0)
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=.01)

show_regression(x, y, model, f'Initial')

for e in range(200):
    train_loss = .0
    for i, (data, label) in enumerate(loader):
        data, label = data.cuda(0), label.cuda(0)
        # print(data)
        optimizer.zero_grad()
        pred = model(data)
        loss = loss_func(pred, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=.75)
        optimizer.step()
        train_loss += loss.cpu().item()
    show_regression(x, y, model, epoch=e + 1)
    print(f'Epoch: {e + 1}/{200}, Training Loss: {train_loss / (i + 1):.4f}')

