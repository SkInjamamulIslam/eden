"""EDEN file 4/7 : eden_model.py  (full EDEN model, free-bits ELBO)"""
import torch
import torch.nn as nn

from eden_encoder import EDENEncoder
from eden_decoder import EDENDecoder
from eden_zinb import ZINBLoss


class EDENModel(nn.Module):
    def __init__(self, n_asvs, n_latent=10,
                 enc_hidden=(256, 128), dec_hidden=(128, 256), dropout=0.1):
        super().__init__()
        self.n_latent = n_latent
        self.encoder = EDENEncoder(n_asvs, n_latent, enc_hidden, dropout)
        self.decoder = EDENDecoder(n_latent, n_asvs, dec_hidden)
        self.loss_fn = ZINBLoss()

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
        total = recon + kl_weight * kl
        return total, recon, kl

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
    n_s, n_a, n_l = 12, 200, 10
    model = EDENModel(n_asvs=n_a, n_latent=n_l)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    x = torch.randint(0, 300, (n_s, n_a)).float()
    s = x.sum(dim=1)
    model.train()
    total, recon, kl = model.elbo(x, s, kl_weight=0.5)
    total.backward()
    assert torch.isfinite(total)
    lat = model.get_latent(x)
    rec = model.get_reconstructed(x, s)
    assert lat.shape == (n_s, n_l) and rec.shape == (n_s, n_a)
    assert (rec >= 0).all()
    print(f"eden_model OK  params {n_par:,} | ELBO {total.item():.4f}")
