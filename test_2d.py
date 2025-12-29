import numpy as np
import time
import torch
from types import SimpleNamespace
from flow_models import *
from flow_models import Flow_based_Bayesian_Filter
from torch.utils.data import TensorDataset
import pandas as pd
import os
import matplotlib.pyplot as plt
import logging
import warnings
from compute import *
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
warnings.filterwarnings("ignore")


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
T = 100  # 时间步数/序列长度
N_train = 1000  # 训练轨迹数
N_test = 200  # 测试轨迹数
dim_x = 2; dim_y = 2

q2 = 0.1
q = torch.sqrt(torch.tensor(q2))  # 过程噪声
r2 = 0.025
r = torch.sqrt(torch.tensor(r2))
K = 150
df1 = pd.read_csv("Data/synthetic_nonlinear/X_train_q={}.csv".format(q2))
x_train = torch.tensor(df1.values, dtype=torch.float32, device=device).reshape(1000, T, N)
y_train = generate_y_from_x_seq(x_train, r=r2, dim_y=dim_y)
df2 = pd.read_csv("Data/synthetic_nonlinear/X_test_q={}.csv".format(q2))
x_test = torch.tensor(df2.values, dtype=torch.float32, device=device).reshape(200, K, N)
y_test = generate_y_from_x_seq(x_test, r=r2, dim_y=dim_y)
m_0=torch.ones(dim_x)

batch_size_train = 64
batch_size_test = 64
train_loader = torch.utils.data.DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size_train,
                                           shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size_test,
                                           shuffle=True, drop_last=True)
arch_params = {"Tx_units": 64, "Tx_layers": 6, "Ty_units": 64, "Ty_layers": 6,
               "A_net_units": 64, "A_net_layers": 6, "B_net_units": 64, "B_net_layers": 6}
model = Flow_based_Bayesian_Filter(arch_params, dim_x, dim_y, train_loader, test_loader, device)
model_dir = "PI/synthetic nonlinear/PI-FBF_r2={}_q={}".format(r2, q2)
model.load_model(model_dir, device)
model_name = "PI-FBF"
model.load_model(model_dir, device)


N1 = 2
results_mmd = []
results_crps = []
rmse_list = []
rmse_list_2=[]
gamma_param = 1/(2*2**2)
for idx_test in range(N1):
    print(idx_test)
    ensemble_size = 500
    measurement = y_test[idx_test].to(device)

    SAD = SimpleNamespace()
    SAD.sigma = q2
    SAD.dt = 1/100
    SAD.Nt = K

    x_ensemble_sample = model.calc_ensemble(ensemble_size,m_0, measurement, SAD, device)# [T-1,500,2]
    x_rmse = torch.sqrt(((x_test[idx_test, 1:,0] - x_ensemble_sample.mean(1)[:,0]) ** 2).mean())

    rmse_list.append(x_rmse)
    x_rmse_2 = torch.sqrt(((x_test[idx_test, 101:,0] - x_ensemble_sample.mean(1)[100:,0]) ** 2).mean())
    rmse_list_2.append(x_rmse_2)

    for k in range(K - 1):
        current_particles = x_ensemble_sample[k]  # [N_particles, dim_u], tensor
        current_truth = x_test[idx_test, k + 1, :]  # [dim_u], tensor
        current_truth_reshaped = current_truth.unsqueeze(0)  # [1, dim_u]

        # --- MMD ---
        mmd_value = calculate_mmd(current_particles, current_truth_reshaped, kernel_gamma=gamma_param)
        mmd_value = torch.as_tensor(mmd_value, device=device, dtype=torch.float32)
        results_mmd.append(mmd_value)

        crps_components = []
        for d in range(dim_x):
            particles_d = current_particles[:, d]
            truth_d = current_truth[d]
            crps_d_value = calculate_crps(particles_d, truth_d)
            crps_components.append(crps_d_value)

        crps_value = torch.mean(torch.stack([torch.as_tensor(v, device=device, dtype=torch.float32)
                                             for v in crps_components]))
        results_crps.append(crps_value)
# --- 计算均值和标准差 ---
print(model_dir)
RMSE = torch.stack(rmse_list)
MMD = torch.stack(results_mmd)
CRPS = torch.stack(results_crps)
RMSE_mean = RMSE.mean()
RMSE_std = RMSE.std(unbiased=True)
print('RMSE mean:', RMSE_mean.item(), 'std:', RMSE_std.item())

RMSE_2 = torch.stack(rmse_list_2)
RMSE_mean_2 = RMSE_2.mean()
RMSE_std_2 = RMSE_2.std(unbiased=True)
print(' 外推时刻的 RMSE mean:', RMSE_mean_2.item(), 'std:', RMSE_std_2.item())


MMD_mean = torch.mean(MMD)
MMD_std = MMD.std(unbiased=True)

CRPS_mean = CRPS.mean()
CRPS_std = CRPS.std(unbiased=True)

print('MMD mean:', MMD_mean.item(), 'std:', MMD_std.item())
print('CRPS mean:', CRPS_mean.item(), 'std:', CRPS_std.item())


def visualize_two_dim_ci(true_state, x_ensemble_sample, k_sigma=3, y_min=None, title=None):
    """
    true_state: [T, 2]  (torch 或 numpy)
    x_ensemble_sample: [T, N_ens, 2]  (torch 或 numpy)
    k_sigma: 置信带倍数，默认 ±3σ
    y_min: 若不为 None，则两幅子图 y 轴下限固定为 y_min
    """
    # 转成 numpy
    ts = true_state.detach().cpu().numpy() if isinstance(true_state, torch.Tensor) else np.asarray(true_state)
    ens = x_ensemble_sample.detach().cpu().numpy() if isinstance(x_ensemble_sample, torch.Tensor) else np.asarray(x_ensemble_sample)

    T, N, D = ens.shape
    assert D == 2, f"最后一维应为2个状态维度，当前为 {D}"
    assert ts.shape[0] == T and ts.shape[1] == 2, f"true_state 形状应为 [T,2]，当前为 {ts.shape}"

    # 均值与无偏标准差（按 ensemble 维度计算）
    mean = ens.mean(axis=1)                 # [T, 2]
    std  = ens.std(axis=1, ddof=1)          # [T, 2]  ddof=1 -> 无偏估计

    lb = mean - k_sigma * std
    ub = mean + k_sigma * std
    t = np.arange(T)
    y_min=-0.5;ymax=2.0
    # fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    for i, ax in enumerate(axes):

        ax.plot(t, ts[:, i], label='Truth', linewidth=1.8, color='orange')
        ax.plot(t, mean[:, i], label='Ensemble mean', linewidth=1.6)
        ax.fill_between(t, lb[:, i], ub[:, i], alpha=0.25, label=f'±{k_sigma}σ')
        ax.set_ylabel(fr'$x_{{{i+1}}}$', fontsize=13)
        ax.axvline(x=100, linestyle='--', color='k', linewidth=1)
        if y_min is not None:
            # 只固定下限，上限自适应
            ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=y_min, top=max(ymax, (ub[:, i].max() if np.isfinite(ub[:, i]).all() else ymax)))
        ax.grid(alpha=0.25)
        ax.legend(loc='upper left', frameon=False)
    title = f"{model_name}  Pred"
    axes[-1].set_xlabel('Time step', fontsize=12)
    if title:
        fig.suptitle(title,fontsize=14)
    plt.show()


SAD = SimpleNamespace()
SAD.sigma = q2
SAD.dt = 1/100
SAD.Nt = K
idx_test = 0
ensemble_size = 500
measurement = y_test[idx_test, 0:K].to(device)
x_ensemble_sample = model.calc_ensemble(ensemble_size, measurement, SAD, device)
x_ensemble_sample = x_ensemble_sample.cpu().detach().numpy()

# evaluate
x_true_value = x_test[idx_test, 1:K]

visualize_two_dim_ci(x_true_value, x_ensemble_sample, k_sigma=3)

print(x_ensemble_sample.shape)