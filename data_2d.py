import torch
import pandas as pd
import os
import numpy as np

def generate_trajectory(K, q, r, dim_x=2, dim_y=2, device="cpu"):

    x = torch.zeros((K, dim_x), device=device)
    y = torch.zeros((K, dim_y), device=device)

    #  x0 ~ N([1,1], 0.1 I_2)
    mean = torch.tensor([1.0, 1.0], device=device)
    cov = 0.1 * torch.eye(dim_x, device=device)
    mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
    x[0] = mvn.sample()

    for k in range(1, K):
        x_prev = x[k-1]

        x_k = 0.9 * torch.sin(1.1 * x_prev + 0.1 * torch.pi) + 0.01
        x_k = x_k + torch.normal(mean=0.0, std=q, size=(dim_x,), device=device)
        x[k] = x_k

        arctan_val = torch.atan(x_k[-1] / x_k[0])
        noise = torch.normal(mean=0.0, std=r, size=(dim_y,), device=device)
        y[k] = arctan_val + noise

    return x, y


dim_x = 2
dim_y = 2
K = 100
N_train = 1000
N_test = 200

q2 = 1
q = torch.sqrt(torch.tensor(q2))
r2 = 0.1
r = torch.sqrt(torch.tensor(r2))  # 0.025,0.05,0.075,0.1

X_train, Y_train = [], []
for _ in range(N_train):
    x, y = generate_trajectory(K, q, r)
    X_train.append(x)
    Y_train.append(y)

X_train = torch.stack(X_train)
Y_train = torch.stack(Y_train)
x_2d = X_train.reshape(N_train, -1).numpy()
y_2d = Y_train.reshape(N_train, -1).numpy()

os.makedirs("Data/synthetic_nonlinear", exist_ok=True)


pd.DataFrame(x_2d).to_csv(f"Data/synthetic_nonlinear/X_train_q={q2}.csv", index=False)
pd.DataFrame(y_2d).to_csv(f"Data/synthetic_nonlinear/Y_train_r2={r2}_q={q2}.csv", index=False)

K2 = 150
X_test, Y_test = [], []

for _ in range(N_test):
    x, y = generate_trajectory(K2, q, r)
    X_test.append(x)
    Y_test.append(y)


X_test = torch.stack(X_test)
Y_test = torch.stack(Y_test)
x_test_2d = X_test.reshape(N_test, -1).numpy()
y_test_2d = Y_test.reshape(N_test, -1).numpy()

pd.DataFrame(x_test_2d).to_csv(f"Data/synthetic_nonlinear/X_test_q={q2}.csv", index=False)
pd.DataFrame(y_test_2d).to_csv(f"Data/synthetic_nonlinear/Y_test_r2={r2}_q={q2}.csv", index=False)



