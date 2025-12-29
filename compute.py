import  torch
def calculate_crps(particles, truth):

    N = len(particles)
    term1 = torch.abs(particles - truth).mean()
    particles_col = particles.view(-1, 1)
    particles_row = particles.view(1, -1)
    diff_matrix = torch.abs(particles_col - particles_row)
    term2 = diff_matrix.sum() / (2.0 * N * N)
    crps_score = term1 - term2
    return crps_score.item()


def rbf_kernel(x, y, gamma):

    x_norm = (x ** 2).sum(dim=1).view(-1, 1)
    y_norm = (y ** 2).sum(dim=1).view(1, -1)
    dist_sq = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    return torch.exp(-gamma * dist_sq)


def calculate_mmd(particles, x_true, kernel_gamma):

    N = particles.shape[0]
    K_pp = rbf_kernel(particles, particles, gamma=kernel_gamma)
    term1 = K_pp.sum() / (N * N)

    K_pt = rbf_kernel(particles, x_true, gamma=kernel_gamma)
    term2 = -2 * K_pt.sum() / N
    term3 = rbf_kernel(x_true, x_true, gamma=kernel_gamma).item()

    mmd_sq = term1 + term2 + term3
    return torch.clamp(mmd_sq, min=0)



