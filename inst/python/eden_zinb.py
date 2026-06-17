"""EDEN file 3/7 : eden_zinb.py"""
import torch
import torch.nn as nn


class ZINBLoss(nn.Module):
    def forward(self, x, mu, theta, pi, eps=1e-8):
        x = x.float()
        log_theta_mu = torch.log(theta + mu + eps)
        nb_zero = theta * (torch.log(theta + eps) - log_theta_mu)
        nb_x = (
            torch.lgamma(x + theta)
            - torch.lgamma(theta + eps)
            - torch.lgamma(x + 1.0)
            + theta * torch.log(theta + eps)
            - theta * log_theta_mu
            + x * torch.log(mu + eps)
            - x * log_theta_mu
        )
        nb_zero_prob = torch.exp(torch.clamp(nb_zero, max=0.0))
        log_p_zero = torch.log(pi + (1.0 - pi) * nb_zero_prob + eps)
        log_p_nonzero = torch.log(1.0 - pi + eps) + nb_x
        log_p = torch.where(x < 0.5, log_p_zero, log_p_nonzero)
        return -log_p.mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    loss_fn = ZINBLoss()
    x = (torch.rand(32, 200) > 0.75).float() * torch.randint(1, 400, (32, 200)).float()
    mu = torch.rand(32, 200) * 40 + 1
    theta = torch.rand(32, 200) * 4 + 0.5
    pi = torch.rand(32, 200) * 0.5
    loss = loss_fn(x, mu, theta, pi)
    assert torch.isfinite(loss) and loss.item() > 0
    print("eden_zinb OK  loss", round(loss.item(), 4))
