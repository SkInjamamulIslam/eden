"""
eden_train_api.py - thin API the R package calls via reticulate.
Wraps eden_model / eden_p_model into one train_eden() function that
returns latent vectors, uncertainty, reconstructed counts and theta.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from eden_model import EDENModel
from eden_p_model import EDENPModel


def _auto_arch(n):
    if n >= 100:
        return (256, 128), (128, 256)
    if n >= 30:
        return (128, 64), (64, 128)
    if n >= 10:
        return (64, 32), (32, 64)
    return (32, 16), (16, 32)


def train_eden(counts, libs, model="EDEN", n_latent=10, epochs=400,
               warmup=100, lr=1e-3, batch=32, free_bits=0.5, max_beta=0.5,
               seed=42, kernel=None):
    """
    counts : (n_samples, n_asvs) float32 numpy array
    libs   : (n_samples,) float32 numpy array
    kernel : (n_asvs, n_asvs) float32, required when model == 'EDEN-P'
    Returns dict of numpy arrays: latent, uncertainty, reconstructed, theta.
    """
    counts = np.asarray(counts, dtype=np.float32)
    libs = np.asarray(libs, dtype=np.float32)
    n, k = counts.shape
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    enc_h, dec_h = _auto_arch(n)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model == "EDEN":
        net = EDENModel(k, int(n_latent), enc_h, dec_h).to(device)
    elif model == "EDEN-P":
        if kernel is None:
            raise ValueError("EDEN-P requires a phylogenetic kernel")
        net = EDENPModel(k, int(n_latent), enc_h, dec_h).to(device)
        net.set_phylo_kernel(np.asarray(kernel, dtype=np.float32))
        net = net.to(device)
    else:
        raise ValueError("model must be 'EDEN' or 'EDEN-P'")

    X = torch.tensor(counts).to(device)
    S = torch.tensor(libs).to(device)
    loader = DataLoader(TensorDataset(X, S),
                        batch_size=min(int(batch), n), shuffle=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-8)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(epochs), eta_min=lr * 0.01)

    best, best_state = float("inf"), None
    for ep in range(1, int(epochs) + 1):
        beta = min(max_beta, (ep - 1) / max(int(warmup), 1) * max_beta)
        net.train()
        er, nb = 0.0, 0
        for xb, sb in loader:
            opt.zero_grad()
            total, recon, kl = net.elbo(xb, sb, kl_weight=beta, free_bits=free_bits)
            total.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            er += recon.item(); nb += 1
        sched.step()
        if ep > int(warmup) and er / nb < best:
            best = er / nb
            best_state = {kk: vv.detach().clone() for kk, vv in net.state_dict().items()}
    if best_state is None:
        best_state = {kk: vv.detach().clone() for kk, vv in net.state_dict().items()}
    net.load_state_dict(best_state)
    net = net.cpu()

    Xc, Sc = X.cpu(), S.cpu()
    latent = net.get_latent(Xc)
    unc = net.get_uncertainty(Xc)
    recon = net.get_reconstructed(Xc, Sc)
    net.eval()
    with torch.no_grad():
        mu_z, _ = net.encoder(Xc)
        _, theta, _, _ = net.decoder(mu_z, Sc)
    theta_mean = theta.mean(dim=0).cpu().numpy()

    return {
        "latent": np.asarray(latent, dtype=np.float64),
        "uncertainty": np.asarray(unc, dtype=np.float64),
        "reconstructed": np.asarray(recon, dtype=np.float64),
        "theta": np.asarray(theta_mean, dtype=np.float64),
        "best_recon": float(best),
    }


if __name__ == "__main__":
    # self-test on tiny synthetic data
    rng = np.random.default_rng(0)
    counts = (rng.random((20, 30)) > 0.7) * rng.integers(1, 200, (20, 30))
    libs = counts.sum(1)
    out = train_eden(counts, libs, model="EDEN", n_latent=4, epochs=60, warmup=20)
    assert out["latent"].shape == (20, 4)
    assert out["reconstructed"].shape == (20, 30)
    assert out["theta"].shape == (30,)
    print("eden_train_api OK | latent", out["latent"].shape,
          "| best_recon", round(out["best_recon"], 4))

    C = np.eye(30, dtype=np.float32)
    out2 = train_eden(counts, libs, model="EDEN-P", n_latent=4,
                      epochs=60, warmup=20, kernel=C)
    assert out2["latent"].shape == (20, 4)
    print("eden_train_api EDEN-P OK | latent", out2["latent"].shape)
