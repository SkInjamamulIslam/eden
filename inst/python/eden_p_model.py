"""EDEN file 5/7 : eden_p_model.py  (phylogenetic kernel smoothing)"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eden_encoder import EDENEncoder
from eden_zinb import ZINBLoss


def build_phylo_kernel(cophenetic_distance, sigma2=1.0, length_scale="median"):
    D = np.asarray(cophenetic_distance, dtype=np.float64)
    nonzero = D[D > 0]
    if length_scale == "median":
        l = float(np.median(nonzero)) if nonzero.size > 0 else 1.0
    else:
        l = float(length_scale)
    C = sigma2 * np.exp(-D / l)
    np.fill_diagonal(C, sigma2)
    ev = np.linalg.eigvalsh(C)
    if ev.min() < 1e-6:
        C = C + (abs(ev.min()) + 1e-4) * np.eye(C.shape[0])
    return C.astype(np.float32), l


class EDENPDecoder(nn.Module):
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
        self.register_buffer("C_phylo", torch.eye(n_asvs))

    def set_kernel(self, C_np):
        C = torch.tensor(np.asarray(C_np, dtype=np.float32))
        assert C.shape == (self.n_asvs, self.n_asvs)
        self.C_phylo = C

    def forward(self, z, s):
        h = self.body(z)
        rho = F.softmax(self.fc_rho(h), dim=-1)
        h_theta = self.fc_theta(h)
        h_mixed = torch.matmul(h_theta, self.C_phylo.t())
        theta = F.softplus(h_mixed) + 1e-4
        pi = torch.sigmoid(self.fc_pi(h))
        mu = s.unsqueeze(1) * rho
        return mu, theta, pi, rho


class EDENPModel(nn.Module):
    def __init__(self, n_asvs, n_latent=10,
                 enc_hidden=(256, 128), dec_hidden=(128, 256), dropout=0.1):
        super().__init__()
        self.n_latent = n_latent
        self.encoder = EDENEncoder(n_asvs, n_latent, enc_hidden, dropout)
        self.decoder = EDENPDecoder(n_latent, n_asvs, dec_hidden)
        self.loss_fn = ZINBLoss()

    def set_phylo_kernel(self, C_np):
        self.decoder.set_kernel(C_np)

    def reparameterise(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x, s):
        mu_z, logvar_z = self.encoder(x)
        z = self.reparameterise(mu_z, logvar_z)
        mu_x, theta, pi, rho = self.decoder(z, s)
        return {"mu_z": mu_z, "logvar_z": logvar_z, "z": z,
                "mu_x": mu_x, "theta": theta, "pi": pi, "rho": rho}

    def elbo(self, x, s, kl_weight=1.0, free_bits=0.5):
        out = self.forward(x, s)
        recon = self.loss_fn(x, out["mu_x"], out["theta"], out["pi"])
        kl_per_dim = -0.5 * (
            1.0 + out["logvar_z"] - out["mu_z"].pow(2) - out["logvar_z"].exp()
        ).mean(dim=0)
        kl = torch.clamp(kl_per_dim, min=free_bits).sum()
        return recon + kl_weight * kl, recon, kl

    @torch.no_grad()
    def get_latent(self, x):
        self.eval()
        mu_z, _ = self.encoder(x)
        return mu_z.cpu().numpy()

    @torch.no_grad()
    def get_uncertainty(self, x):
        self.eval()
        _, logvar_z = self.encoder(x)
        return torch.exp(0.5 * logvar_z).cpu().numpy()

    @torch.no_grad()
    def get_reconstructed(self, x, s):
        self.eval()
        mu_z, _ = self.encoder(x)
        mu_x, _, _, _ = self.decoder(mu_z, s)
        return mu_x.cpu().numpy()
        
    @torch.no_grad()
    def get_theta(self, x, s):
        self.eval()
        mu_z, _ = self.encoder(x)
        _, theta, _, _ = self.decoder(mu_z, s)
        return theta.mean(dim=0).cpu().numpy()

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    n_a = 60
    raw = np.random.exponential(0.5, (n_a, n_a))
    D = (raw + raw.T) / 2
    np.fill_diagonal(D, 0.0)
    C, l = build_phylo_kernel(D)
    assert np.linalg.eigvalsh(C).min() > -1e-6
    model = EDENPModel(n_asvs=n_a, n_latent=8)
    model.set_phylo_kernel(C)
    x = torch.randint(0, 200, (12, n_a)).float()
    s = x.sum(dim=1)
    model.train()
    total, recon, kl = model.elbo(x, s, kl_weight=0.3)
    total.backward()
    assert torch.isfinite(total)
    print(f"eden_p_model OK  kernel {C.shape} l={l:.3f} | ELBO {total.item():.4f}")
