"""EDEN file 2/7 : eden_decoder.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EDENDecoder(nn.Module):
    def __init__(self, n_latent, n_asvs, hidden=(128, 256)):
        super().__init__()
        layers = []
        prev = n_latent
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.LeakyReLU(0.1))
            prev = h
        self.body = nn.Sequential(*layers)
        self.fc_rho = nn.Linear(prev, n_asvs)
        self.fc_theta = nn.Linear(prev, n_asvs)
        self.fc_pi = nn.Linear(prev, n_asvs)
        self.n_asvs = n_asvs

    def forward(self, z, s):
        h = self.body(z)
        rho = F.softmax(self.fc_rho(h), dim=-1)
        theta = F.softplus(self.fc_theta(h)) + 1e-4
        pi = torch.sigmoid(self.fc_pi(h))
        mu = s.unsqueeze(1) * rho
        return mu, theta, pi, rho


if __name__ == "__main__":
    torch.manual_seed(0)
    dec = EDENDecoder(n_latent=10, n_asvs=200)
    dec.eval()
    z = torch.randn(8, 10)
    s = torch.randint(5000, 50000, (8,)).float()
    mu, theta, pi, rho = dec(z, s)
    assert torch.allclose(rho.sum(dim=1), torch.ones(8), atol=1e-4)
    assert (theta > 0).all()
    assert (pi >= 0).all() and (pi <= 1).all()
    assert torch.allclose(mu.sum(dim=1), s, rtol=1e-3)
    print("eden_decoder OK  rho_sum", round(rho.sum(1).mean().item(), 5))
