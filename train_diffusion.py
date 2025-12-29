import torch

import pandas as pd
from torch.func import jacrev,vmap
from torch.optim.lr_scheduler import LambdaLR
from flow_models import *
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import logging
from tqdm import tqdm
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
torch.manual_seed(1226)
np.random.seed(1226)
torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_observations(u, n_sensors, r):
    num_traj, T, Ns = u.shape
    sensor_idx = torch.linspace(0, Ns - 1, n_sensors).long()
    u_obs = u[:, :, sensor_idx]
    noise = torch.randn_like(u_obs) * torch.sqrt(torch.tensor(r, dtype=u.dtype))
    Y = torch.exp(-u_obs + 0.5) + noise
    return Y, sensor_idx


r2 = 1
# r = torch.sqrt(torch.tensor(r2))  # 0.25,0.5,0.75,1.0 观测噪声：在不同强度观测噪声的情况下测试方法
N = 15; T = 100
dim_u = 50; dim_y = N; num_layers=6; hidden_dim=64
m = 1
x = torch.linspace(-1.0, 1.0, 100 + 2, device=device)[1:-1][::int(100/dim_u)]
D = 0.1     # 扩散系数
kappa = 1   # 对流系数
delta_t = torch.tensor(1/T, dtype=torch.float32, device=device)
sigma = 1
Sigma = sigma*torch.sqrt(delta_t)

df1 = pd.read_csv("Data/diffusion_equation/u_train_Ns={}_T={}_sigma={}.csv".format(dim_u, T,sigma))
U_train = torch.tensor(df1.values, dtype=torch.float32, device=device).reshape(1000, T, dim_u)
Y_train, _ = generate_observations(U_train, n_sensors=N, r=r2)
df2 = pd.read_csv("Data/diffusion_equation/u_test_Ns={}_T={}_sigma={}.csv".format(dim_u, T, sigma))
U_test = torch.tensor(df2.values, dtype=torch.float32, device=device).reshape(200, T, dim_u)
Y_test, _ = generate_observations(U_test, n_sensors=N, r=r2)

batch_size = 128
loader_train = torch.utils.data.DataLoader(TensorDataset(U_train, Y_train), batch_size=batch_size,
                                           shuffle=True, drop_last=True)
loader_test = torch.utils.data.DataLoader(TensorDataset(U_test, Y_test), batch_size=8,
                                           shuffle=True, drop_last=True)
arch_params = {"Tx_units": 64, "Tx_layers": 6, "Ty_units": 64, "Ty_layers": 6,
               "A_net_units": 64, "A_net_layers": 6, "B_net_units": 64, "B_net_layers": 6}

model = Flow_based_Bayesian_Filter(arch_params, dim_u, dim_y, loader_train, loader_test, device)

num_epochs = 1000
lr = 5e-4; momentum = 0.9; decay = 0.99; final_decay=5e-2
trainable_params = model.params_NFs + model.A_net_params + model.B_net_params + \
                           [model.Px, model.Py, model.C, model.D]
optimizer = optim.Adam(trainable_params, lr=lr, betas=(momentum, decay), eps=1e-7)
gamma = (final_decay)**(1./num_epochs)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

model_dir2 = "PI/diffusion_equation/r2={}/PI_FBF_sensor={}_{}_{}_sigma={}".format(r2, N, dim_u, T, sigma)
os.makedirs(model_dir2, exist_ok=True)  # 创建目录，如果已经存在则忽略

pbar1 = tqdm(total=num_epochs//m, desc="Steps")

for epoch in range(num_epochs):
    loss1 = 0
    for batch_idx, (u, y) in enumerate(loader_train):
        batch_idx += 1
        B = u.shape[0]
        u, y = u.float().to(device), y.float().to(device)
        u_old = u[:, 0:-1]
        u_new, y_new = u[:, 1:], y[:, 1:]
        loss_state, loss_obs, loss = model.running_steps(u_old, u_new, y_new)
        loss1 += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  # 清空梯度，为下一次迭代做准备
    loss1 = loss1 / batch_idx
    scheduler.step()

    if epoch % m == 0:
        logging.info("\nCompleted step {}:".format(epoch))
        pbar1.set_postfix(
            # loss=f"{loss:.8f}",  # 直接使用 float 变量 loss
            loss_MAP=f"{loss1:.8f}",  # loss1 仍然是张量,所以需要 .item()
            )
        pbar1.update(1)  # 更新进度条

batch_size2 = 8
loader_train2 = torch.utils.data.DataLoader(TensorDataset(U_train, Y_train), batch_size=batch_size2,
                                           shuffle=True, drop_last=True)
num_epochs2 = 3
optimizer2 = optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-2)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=int(len(loader_train2)/4), gamma=0.9)
step = 0
pbar2 = tqdm(total=num_epochs2*len(loader_train2)//m, desc="Steps")
for epoch in range(num_epochs2):
    # weight_loss2 = min(0.5, 0.1 + 0.1 * epoch)
    for batch_idx, (u, y) in enumerate(loader_train2):

        B = u.shape[0]
        u, y = u.float().to(device), y.float().to(device)
        u_old = u[:, 0:-1]
        u_new, y_new = u[:, 1:], y[:, 1:]
        loss_state, loss_obs, loss = model.running_steps(u_old, u_new, y_new)
        loss1 = loss


        def iteration(u_0, y1):
            U_0, _ = model.Tx(u_0)
            U_1 = model.A_net(y1) + torch.matmul(model.B_net(y1).view(-1, dim_u, dim_u),
                                                      U_0.unsqueeze(-1)).squeeze(-1)
            u_1, _ = model.Tx(U_1, mode='inverse')
            return u_1.squeeze()


        func1 = jacrev(iteration)
        func2 = lambda u, y: jacrev(jacrev(lambda uu: iteration(uu, y)))(u)
        idx = torch.arange(dim_u, device=device)

        J0 = torch.diag(-np.pi * torch.cos(np.pi * x)).unsqueeze(0).repeat(B, 1, 1)  # [B, N, N]
        H_slice = torch.zeros(B, dim_u, dim_u, device=device, dtype=u.dtype)
        H_slice[:, idx, idx] = (np.pi ** 2) * torch.sin(np.pi * x)  # H[b,i,i] = π² sin(π s[i])

        K = 1  # 每 K 步截断一次
        loss2 = 0.0
        u0 = u_old[:, 0]  # [B, N]
        if step%10 == 0:
            weight_loss2 = min(5, 0.02 * step)
        for start in range(0, T - 1, K):
            end = min(start + K, T - 1)
            loss_chunk = []
            for t in range(start, end):
                J0_1 = vmap(func1)(u0, y_new[:, t])  # [B, out=N, in=N]  (b, p, t) 用 p 表示输出 index
                H0_1 = vmap(func2)(u0, y_new[:, t])  # [B, p, m, n]  对应 H0_1[b,p,m,n] = ∂^2 u1_p / ∂u0_m ∂u0_n

                termA_slice = torch.einsum('bpm,bmi->bpi', J0_1, H_slice)  # [B, p, i]
                termB_slice = torch.einsum('bpmn,bmi,bni->bpi', H0_1, J0, J0)  # [B, p, i]

                H_slice = termA_slice + termB_slice  # [B, p, i]squeue
                J1 = torch.einsum('bij,bjk->bik', J0_1, J0)  # [B, N, N]
                H1_diag_diag = H_slice[:,  idx, idx]  # [B, N]
                J1_diag = torch.diagonal(J1, dim1=1, dim2=2)  # [B, N]
                u1 = iteration(u0, y_new[:, t])
                res_t = (u1 - u0) + delta_t*(kappa * J1_diag - D * H1_diag_diag - 5 * (x ** 2 - 1))
                loss_pde = torch.mean(res_t ** 2) / (2 * Sigma ** 2)
                J0 = J1
                u0 = u1
                loss_chunk.append(loss_pde)
                # u0 = u_new[:, t]

            loss_chunk = torch.stack(loss_chunk).sum()
            # loss_chunk = loss_chunk/u_old.shape[0]/K
            retain_flag = (end < T - 1)
            (loss_chunk * weight_loss2 / (end - start)).backward(retain_graph=retain_flag)
            loss2 += loss_chunk.item()

            J0 = J0.detach()
            H_slice = H_slice.detach()
            u0 = u0.detach()

        loss1.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=10.0)
        optimizer2.step()
        optimizer2.zero_grad()  # 清空梯度，为下一次迭代做准备
        step += 1
        if step % m == 0:
            logging.info("\nCompleted step {}:".format(step))
            pbar2.set_postfix(
                # loss=f"{loss:.8f}",  # 直接使用 float 变量 loss
                loss_MAP=f"{loss1.item():.8f}",  # loss1 仍然是张量,所以需要 .item()
                loss_PDE=f"{loss2:.8f}",  # 直接使用 float 变量 loss2
            )
            pbar2.update(1)  # 更新进度条
    scheduler2.step()

    model.save_model(model_dir2)
