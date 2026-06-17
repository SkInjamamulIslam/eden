"""EDEN file 1/7 : eden_encoder.py"""
import torch
import torch.nn as nn


class EDENEncoder(nn.Module):
    def __init__(self, n_asvs, n_latent=10, hidden=(256, 128), dropout=0.1):
        super().__init__()
        layers = []
        prev = n_asvs
        for i, h in enumerate(hidden):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.LeakyReLU(0.1))
            if i == 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.body = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, n_latent)
        self.fc_lv = nn.Linear(prev, n_latent)
        self.n_latent = n_latent

    def forward(self, x):
        h = self.body(torch.log1p(x))
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_lv(h), -10.0, 10.0)
        return mu, logvar


if __name__ == "__main__":
    torch.manual_seed(0)
    enc = EDENEncoder(n_asvs=200, n_latent=10)
    enc.eval()
    x = torch.randint(0, 400, (8, 200)).float()
    mu, lv = enc(x)
    assert mu.shape == (8, 10)
    assert lv.shape == (8, 10)
    print("eden_encoder OK  mu", tuple(mu.shape), "logvar", tuple(lv.shape))
