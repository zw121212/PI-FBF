from types import SimpleNamespace

import torch
from torch.utils.data import TensorDataset
import pandas as pd
from flow_models import *
import matplotlib.pyplot as plt

from compute import *
import warnings
warnings.filterwarnings("ignore")


def generate_observations(u, n_sensors, r):
    num_traj, T, Ns = u.shape
    sensor_idx = torch.linspace(0, Ns - 1, n_sensors).long()
    u_obs = u[:, :, sensor_idx]
    noise = torch.randn_like(u_obs) * torch.sqrt(torch.tensor(r, dtype=u.dtype))
    Y = torch.exp(-u_obs + 0.5) + noise
    return Y, sensor_idx


sigma = 1
r2 = 0.25
# r = torch.sqrt(torch.tensor(r2))  # 0.25,0.5,1.0
N = 10; T = 100
device = 'cpu'
dim_u = 50; dim_y = N; num_layers=6; hidden_dim=64

df1 = pd.read_csv("Data/diffusion_equation/u_train_Ns={}_T={}_sigma={}.csv".format(dim_u, T,sigma))
U_train = torch.tensor(df1.values, dtype=torch.float32, device=device).reshape(1000, T, dim_u)
Y_train, _ = generate_observations(U_train, n_sensors=N, r=r2)
K = 200
df2 = pd.read_csv("Data/diffusion_equation/u_test_Ns={}_T={}_sigma={}.csv".format(dim_u, K,sigma))
U_test = torch.tensor(df2.values, dtype=torch.float32, device=device).reshape(200, K, dim_u)
K = 150
U_test = U_test[:, :K, :]
Y_test, _ = generate_observations(U_test, n_sensors=N, r=r2)
x1 = torch.linspace(-1, 1,200)[::4]
m_0 = -torch.sin(math.pi * x1)


loader_train = torch.utils.data.DataLoader(TensorDataset(U_train, Y_train), batch_size=128,
                                           shuffle=True, drop_last=True)
loader_test = torch.utils.data.DataLoader(TensorDataset(U_test, Y_test), batch_size=128,
                                           shuffle=True, drop_last=True)
arch_params = {"Tx_units": 64, "Tx_layers": 6, "Ty_units": 64, "Ty_layers": 6,
               "A_net_units": 64, "A_net_layers": 6, "B_net_units": 64, "B_net_layers": 6}
model = Flow_based_Bayesian_Filter(arch_params, dim_u, dim_y, loader_train, loader_test, device)
model_dir = "PI/diffusion_equation/r2={}/PI_FBF_sensor={}_{}_{}_sigma={}".format(r2, N, dim_u, T,sigma)

model.load_model(model_dir, device)
model_name = "PI-FBF"

N1 = 100
mmd_all_sims = torch.zeros((N1, K-1), device=device)
crps_all_sims = torch.zeros((N1, K-1), device=device)
rmse_list = []
rmse_list_2 = []
Pred = torch.zeros((N1,K-1,dim_u))

gamma_param = 1/(2*2**2)
for idx_test in range(N1):
    print(idx_test)
    ensemble_size = 500
    measurement = Y_test[idx_test].to(device)

    SAD = SimpleNamespace()
    SAD.sigma = sigma
    SAD.dt = 1/100
    SAD.Nt = K

    x_ensemble_sample = model.calc_ensemble(ensemble_size, measurement, SAD, device)# [99,500,50]
    Pred[idx_test] = x_ensemble_sample.mean(1)
    x_rmse = torch.sqrt(((U_test[idx_test, 1:100] - x_ensemble_sample.mean(1)[:99,:]) ** 2).mean())


    rmse_list.append(x_rmse)
    x_rmse_2 = torch.sqrt(((U_test[idx_test, 100:] - x_ensemble_sample.mean(1)[99:]) ** 2).mean())
    rmse_list_2.append(x_rmse_2)


    for k in range(K - 1):
        current_particles = x_ensemble_sample[k]  # [N_particles, dim_u], tensor
        current_truth = U_test[idx_test, k + 1, :]  # [dim_u], tensor
        current_truth_reshaped = current_truth.unsqueeze(0)  # [1, dim_u]

        # --- MMD ---
        mmd_value = calculate_mmd(current_particles, current_truth_reshaped, kernel_gamma=gamma_param)
        mmd_value = torch.as_tensor(mmd_value, device=device, dtype=torch.float32)
        # results_mmd.append(mmd_value)
        mmd_all_sims[idx_test, k] = mmd_value
        crps_components = []
        for d in range(dim_u):
            particles_d = current_particles[:, d]
            truth_d = current_truth[d]
            crps_d_value = calculate_crps(particles_d, truth_d)
            crps_components.append(crps_d_value)

        crps_value = torch.mean(torch.stack([torch.as_tensor(v, device=device, dtype=torch.float32)
                                             for v in crps_components]))
        # results_crps.append(crps_value)
        crps_all_sims[idx_test, k] = crps_value

print(model_dir)
RMSE = torch.stack(rmse_list)
RMSE_mean = RMSE.mean()
RMSE_std = RMSE.std(unbiased=True)
print('RMSE mean:', RMSE_mean.item(), 'std:', RMSE_std.item())
MMD = mmd_all_sims.mean(1)
CRPS = crps_all_sims.mean(1)

MMD_mean = torch.mean(MMD)
MMD_std = MMD.std(unbiased=True)

CRPS_mean = CRPS.mean()
CRPS_std = CRPS.std(unbiased=True)

print('MMD mean:', MMD_mean.item(), 'std:', MMD_std.item())
print('CRPS mean:', CRPS_mean.item(), 'std:', CRPS_std.item())
RMSE_2 = torch.stack(rmse_list_2)
RMSE_mean_2 = RMSE_2.mean()
RMSE_std_2 = RMSE_2.std(unbiased=True)
print(' 外推时刻的 RMSE mean:', RMSE_mean_2.item(), 'std:', RMSE_std_2.item())
Error = (Pred.mean(0)-U_test[:N1,1:,:].mean(0)).abs()
Mean_Error = torch.sqrt((Error**2).mean())
print('均值误差',Mean_Error)

Error_1 = (Pred.mean(0)[100:,:]-U_test[:N1,101:,:].mean(0)).abs()
Mean_Error_1 = torch.sqrt((Error_1**2).mean())
print('外推均值误差', Mean_Error_1)

x = np.linspace(-1, 1, dim_u)
t = np.linspace(0, 1.5, K)

# 绘制空间-时间图像
plt.figure(figsize=(7, 4))
plt.imshow(Error.detach().numpy().T, extent=[ t.min(), t.max(), x.min(), x.max()],
           origin='lower', aspect='auto', cmap='turbo',vmax=2)
plt.colorbar(label='error')
plt.ylabel('Space x')
plt.xlabel('Time t')
plt.title(f"{model_name} Mean Error")
plt.tight_layout()
plt.show()
x = np.linspace(-1.0, 1.0, 100 + 2)[1:-1][::2]
sensor_idx = torch.linspace(0, dim_u - 1, N).long()
x1 = x[sensor_idx]


def visualize_confidence_interval_spatial(T, true_state, x_ensemble_sample, x_P_plus):
    indices = [25, 50, 75, 100, 125]
    for idx in indices:
        mean = x_ensemble_sample[idx].mean(0)
        std = x_ensemble_sample[idx].std(0)

        lb = mean - 3 * std
        ub = mean + 3 * std

        plt.style.use('default')
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)

        ax.plot(x, true_state[idx], color="orange", label="Truth")
        ax.plot(x, mean, label=f"{model_name} ")
        ax.fill_between(x, lb, ub, facecolor='deepskyblue', alpha=0.5,
                        label=r'$\pm 3$ Std')
        ax.scatter(x1, true_state[idx][sensor_idx],
                   color="orange", label="sensor")

        ax.set_ylim(bottom=-7)
        ax.legend(fontsize=12, loc='upper right')

        plt.tight_layout()
        plt.show()


def visualize_std_spacetime(x_ensemble_sample):
    std_spacetime = x_ensemble_sample.std(axis=1)
    plt.figure(figsize=(6, 4))
    im = plt.imshow(
        std_spacetime.T,
        extent=[0, 1.5, -1, 1],   # 若 x,t 有真实坐标可替换
        aspect='auto',
        origin='lower',
        cmap='turbo',
        vmax=0.12
    )
    plt.axvline(x=1.0, linestyle='--', color='k', linewidth=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.ylabel("Space x")
    plt.xlabel("Time t")
    plt.title("Std Across Spacetime")
    plt.tight_layout()
    plt.show()


for l in range(5):
# l=2
    x_true_value = U_test[l, 1:]
    measurement = Y_test[l].to(device)
    SAD = SimpleNamespace()
    SAD.sigma = sigma
    SAD.dt = 1/100
    SAD.Nt = K


    def Calc_sample_covariance(x_ensemble_sample):
        ensemble_size = x_ensemble_sample.shape[1]
        x_mean = x_ensemble_sample.mean(axis=1, keepdims=True)
        x_dev = x_ensemble_sample - x_mean
        return 1 / (ensemble_size - 1) * np.matmul(x_dev.transpose(0, 2, 1), x_dev)

    x_ensemble_sample = model.calc_ensemble(1000, m_0, measurement, SAD, device).cpu().detach().numpy()
    # print(x_ensemble_sample.shape)
    visualize_std_spacetime(x_ensemble_sample)
    # x_P_plus = Calc_sample_covariance(x_ensemble_sample)
    # visualize_confidence_interval_spatial(K, x_true_value, x_ensemble_sample, x_P_plus)

    pred = x_ensemble_sample.mean(1).T  # [N=50, T=200]
    true = x_true_value.T.cpu().numpy()               # [N=50, T=200]
    abs_error = np.abs(pred - true)
    t = np.linspace(0, 2, K)[1:]
    # --------- 画图部分 -------------
    fig, axs = plt.subplots(1, 3, figsize=(30, 6))

    # 预测
    im0 = axs[0].imshow(true, extent=[0, 1.5, -1, 1], aspect='auto', origin='lower')
    axs[0].set_title("True")
    axs[0].set_xlabel("Time t")
    axs[0].set_ylabel("Space x")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    # 真实
    im1 = axs[1].imshow(pred, extent=[0, 1.5, -1, 1], aspect='auto', origin='lower')
    axs[1].set_title(f"{model_name}  Pred")
    axs[1].set_xlabel("Time t")
    axs[1].set_ylabel("Space x")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # 绝对误差
    im2 = axs[2].imshow(abs_error, extent=[0, 1.5, -1, 1], aspect='auto', origin='lower', cmap='turbo',vmax=2)
    axs[2].set_title(f"{model_name}  Absolute Error")
    axs[2].set_xlabel("Time t")
    axs[2].set_ylabel("Space x")
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
    t = np.linspace(1, 2, 100)[1:50]
    # fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    #

    # im0 = axs[0].imshow(true[:,100:150], extent=[1, 1.5, -1, 1], aspect='auto', origin='lower')
    # axs[0].set_title("True")
    # axs[0].set_xlabel("Time t")
    # axs[0].set_ylabel("Space x")
    # fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    # im1 = axs[1].imshow(pred[:,100:150], extent=[1, 1.5, -1, 1], aspect='auto', origin='lower')
    # axs[1].set_title(f"{model_name}  Pred")
    # axs[1].set_xlabel("Time t")
    # axs[1].set_ylabel("Space x")
    # fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # im2 = axs[2].imshow(abs_error[:,100:150], extent=[1, 1.5, -1, 1], aspect='auto', origin='lower', cmap='turbo',vmax=2)
    # axs[2].set_title(f"{model_name}  Absolute Error")
    # axs[2].set_xlabel("Time t")
    # axs[2].set_ylabel("Space x")
    # fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    #
    # plt.tight_layout()
    # plt.show()
    # --- 图 1: True ---
    fig, ax = plt.subplots(figsize=(6, 4))
    im0 = ax.imshow(true, extent=[0, 1.5, -1, 1], aspect='auto', origin='lower')
    ax.set_title("True")
    ax.axvline(x=1.0, linestyle='--', color='k', linewidth=1)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Space x")
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots(figsize=(6, 4))
    im1 = ax.imshow(pred, extent=[0, 1.5, -1, 1], aspect='auto', origin='lower')
    ax.set_title(f"{model_name}  Pred")

    ax.axvline(x=1.0, linestyle='--', color='k', linewidth=1)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Space x")
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots(figsize=(6, 4))
    im2 = ax.imshow(abs_error, extent=[0, 1.5, -1, 1],
                    aspect='auto', origin='lower', cmap='turbo', vmax=2)
    ax.set_title(f"{model_name}  Absolute Error")
    ax.axvline(x=1.0, linestyle='--', color='k', linewidth=1)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Space x")
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
