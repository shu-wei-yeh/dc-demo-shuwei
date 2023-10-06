from typing import Optional

import torch

from utils.plotting import plot_psds as plot_psds_


@torch.no_grad()
def plot_psds(
    pred: torch.Tensor,
    strain: torch.Tensor,
    mask: torch.Tensor,
    spectral_density: torch.nn.Module,
    fftlength: float,
    asd: bool = True,
    fname: Optional[str] = None,
):
    mask = mask.cpu().numpy()
    cleaned = strain - pred

    def _plottable_psd(x):
        x = spectral_density(x.double())
        if asd:
            x = x**0.5
        x = torch.quantile(x, 0.5, dim=0)
        return x.cpu().numpy()

    cleaned = _plottable_psd(cleaned)[mask]
    raw = _plottable_psd(strain)[mask]
    pred = _plottable_psd(pred)[mask]

    freqs = torch.arange(len(mask)).cpu().numpy()
    freqs = freqs / fftlength
    freqs = freqs[mask]
    return plot_psds_(freqs, pred, raw, cleaned, asd=asd, fname=fname)
