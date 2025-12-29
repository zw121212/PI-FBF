import numpy as np
import time
import torch
from flow_models import *
from flow_models import Flow_based_Bayesian_Filter
from torch.utils.data import TensorDataset
import pandas as pd
import os
import logging
from tqdm import tqdm
from types import SimpleNamespace
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1226)
np.random.seed(1226)


def generate_y_from_x_seq(X, r, dim_y=2):
    """
    X: [N, K, dim_x]
    return: Y [N, K, dim_y]
    """
    device = X.device
    x1 = X[..., 0]
    x2 = X[..., -1]
    mean = torch.atan2(x2, x1)[..., None]           # [N, K, 1]
    mean = mean.expand(-1, -1, dim_y)               # [N, K, dim_y]
    noise = torch.normal(mean=0.0, std=r, size=(dim_y,), device=device)
    return mean + noise


N = 2
T = 100
N_train = 1000
N_test = 200
dim_x = 2 ; dim_y = 2

q2 = 0.1
q = torch.sqrt(torch.tensor(q2))
r2 = 1
r = torch.sqrt(torch.tensor(r2))

df1 = pd.read_csv("Data/synthetic_nonlinear/X_train_q={}.csv".format(q2))
x_train = torch.tensor(df1.values, dtype=torch.float32, device=device).reshape(1000, T, N)
y_train = generate_y_from_x_seq(x_train, r=r, dim_y=dim_y)
df2 = pd.read_csv("Data/synthetic_nonlinear/X_test_q={}.csv".format(q2))
K = 150
x_test = torch.tensor(df2.values, dtype=torch.float32, device=device).reshape(200, K, N)
y_test = generate_y_from_x_seq(x_test, r=r, dim_y=dim_y)


batch_size_train = 64
batch_size_test = 64
train_loader = torch.utils.data.DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size_train,
                                           shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size_test,
                                           shuffle=True, drop_last=True)
n_epochs = 1000
arch_params = {"Tx_units": 64, "Tx_layers": 6, "Ty_units": 64, "Ty_layers": 6,
               "A_net_units": 64, "A_net_layers": 6, "B_net_units": 64, "B_net_layers": 6}
model = Flow_based_Bayesian_Filter(arch_params, dim_x, dim_y, train_loader, test_loader, device)
model.FBF_compile(n_epochs)

model_dir2 = "PI/synthetic nonlinear/PI-FBF_r2={}_q={}".format(r2, q2)
os.makedirs(model_dir2, exist_ok=True)
# model.load_model(model_dir1, device)
try:
    optimal_loss = float('inf')
    start_time = time.time()

    for epoch in range(n_epochs):
        model.train(epoch, device)
        model.test(epoch, device)

        if optimal_loss > model.loss_test[-1]:
            optimal_loss = model.loss_test[-1]
        model.scheduler.step()

except KeyboardInterrupt:
    pass
finally:
    print("==> Best test loss: %.6f" % (optimal_loss))
    print("Time elapsed:", time.time() - start_time)


batch_size2 = 64
loader_train2 = torch.utils.data.DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size2,
                                           shuffle=True, drop_last=True)
num_epochs2 = 200
trainable_params = model.params_NFs + model.A_net_params + model.B_net_params + \
                           [model.Px, model.Py, model.C, model.D]
optimizer2 = optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-2)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=10*len(loader_train2), gamma=0.9)
step = 0;m = 5
pbar2 = tqdm(total=num_epochs2*len(loader_train2)//m, desc="Steps")
x_rmse = 0
for epoch in range(num_epochs2):
    weight_loss2 = min(0.5, 0.1 + 0.05 * epoch)

    for batch_idx, (x, y) in enumerate(loader_train2):
        B = x.shape[0]
        x, y = x.float().to(device), y.float().to(device)
        x_old = x[:, 0:-1]
        x_new, y_new = x[:, 1:], y[:, 1:]
        loss_state, loss_obs, loss = model.running_steps(x_old, x_new, y_new)
        loss1 = loss

        def iteration(u_0, y1):
            U_0, _ = model.Tx(u_0)
            U_1 = model.A_net(y1) + torch.matmul(model.B_net(y1).view(-1, dim_x, dim_x),
                                                      U_0.unsqueeze(-1)).squeeze(-1)
            u_1, _ = model.Tx(U_1, mode='inverse')
            return u_1.squeeze()

        loss2 = 0.0
        x0 = x_old[:, 0]  # [B, N]
        for t in range(T-1):
            y1 = y_new[:, t]
            x1 = iteration(x0, y1)
            x_target = 0.9 * torch.sin(1.1 * x0 + 0.1 * torch.pi) + 0.01
            loss_function = torch.mean((x_target - x1) ** 2) / (2 * q2)
            loss2 += loss_function
            x0 = x_old[:, t]
            # x0 = x1
        loss = loss1 + weight_loss2 * loss2

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
        optimizer2.step()
        optimizer2.zero_grad()  # 清空梯度，为下一次迭代做准备
        step += 1
        if step % m == 0:
            logging.info("\nCompleted step {}:".format(step))
            pbar2.set_postfix(
                loss=f"{loss:.8f}",  # 直接使用 float 变量 loss
                loss_MAP=f"{loss1.item():.8f}",  # loss1 仍然是张量,所以需要 .item()
                loss_PDE=f"{loss2:.8f}",  # 直接使用 float 变量 loss2
                # x_RMSE=f"{x_rmse:.4f}",
            )
            pbar2.update(1)  # 更新进度条
    scheduler2.step()
    # if (1+epoch)%1 == 0:
    #     idx_test=1
    #     SAD = SimpleNamespace()
    #     SAD.sigma = q2
    #     SAD.dt = 1 / 100
    #     SAD.Nt = K
    #     x_ensemble_sample = model.calc_ensemble(100, y_test[idx_test], SAD, device)  # [99,500,50]
    #     x_rmse = torch.sqrt(((x_test[idx_test, 1:] - x_ensemble_sample.mean(1)) ** 2).mean()).to(device)
        # print(x_rmse)
    if (1 + epoch) % 10 == 0:
        model.save_model(model_dir2)
print(model_dir2)