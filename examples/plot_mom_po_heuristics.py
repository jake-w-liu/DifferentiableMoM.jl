#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
FIGS = ROOT / 'figs'
FIGS.mkdir(parents=True, exist_ok=True)

# 1) Direct MoM vs PO comparison (with aligned PO curve)
cmp_csv = DATA / 'mom_vs_pofacets_direct_compare_interp.csv'
if cmp_csv.exists():
    df = pd.read_csv(cmp_csv).sort_values('theta_deg')
    bias = df['delta_dB'].mean()

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(df['theta_deg'], df['sigma_mom_dBsm'], lw=2, label='MoM')
    ax[0].plot(df['theta_deg'], df['sigma_po_dBsm_interp'], lw=2, ls='--', label='POFacets')
    ax[0].plot(df['theta_deg'], df['sigma_po_dBsm_interp'] + bias, lw=1.8, ls=':', label=f'PO + bias ({bias:+.2f} dB)')
    ax[0].set_ylabel('RCS (dBsm)')
    ax[0].set_title('Heuristic RCS Comparison (phi ~ 5 deg)')
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    ax[1].plot(df['theta_deg'], df['delta_dB'], lw=1.8, color='firebrick', label='MoM - PO')
    ax[1].axhline(0, color='k', lw=1, alpha=0.5)
    ax[1].axhline(bias, color='gray', lw=1, ls='--', alpha=0.8, label=f'bias={bias:.2f} dB')
    ax[1].set_xlabel('theta (deg)')
    ax[1].set_ylabel('Delta (dB)')
    ax[1].grid(alpha=0.3)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(FIGS / 'heuristic_mom_vs_pofacets_overlay.png', dpi=180)

# 2) Variant sensitivity sweep
var_csv = DATA / 'mom_vs_pofacets_variant_metrics.csv'
if var_csv.exists():
    df = pd.read_csv(var_csv).sort_values('rmse_aligned_db')
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(df))
    ax.bar(x, df['rmse_aligned_db'], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df['tag'], rotation=30, ha='right')
    ax.set_ylabel('Aligned RMSE (dB)')
    ax.set_title('Heuristic Sensitivity to PO Setup Conventions')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / 'heuristic_variant_sensitivity.png', dpi=180)

# 3) Frequency trend
freq_csv = DATA / 'mom_vs_pofacets_freq_sweep_metrics.csv'
if freq_csv.exists():
    df = pd.read_csv(freq_csv).sort_values('freq_ghz')
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(df['freq_ghz'], df['rmse_aligned_db'], 'o-', label='RMSE aligned')
    ax.plot(df['freq_ghz'], df['mae_aligned_db'], 's--', label='MAE aligned')
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Error (dB)')
    ax.set_title('MoM vs PO Heuristic Error vs Frequency')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / 'heuristic_freq_trend.png', dpi=180)

# 4) Electrical edge-size diagnostic from airplane.mat
mat_path = ROOT.parent / 'pofacets4.5' / 'pofacets4.5' / 'CAD Library Pofacets' / 'airplane.mat'
if mat_path.exists():
    try:
        from scipy.io import loadmat
        m = loadmat(mat_path)
        coord = m['coord']
        tri = m['facet'][:, :3].astype(int) - 1
        edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        L = np.linalg.norm(coord[edges[:, 0]] - coord[edges[:, 1]], axis=1)

        fig, ax = plt.subplots(figsize=(9, 5))
        for f in [0.1, 0.3, 1.0, 3.0]:
            lam = 299792458.0 / (f * 1e9)
            vals = np.sort(L / lam)
            cdf = np.linspace(0, 1, len(vals), endpoint=True)
            ax.plot(vals, cdf, lw=2, label=f'{f:g} GHz')
        ax.axvline(0.1, color='k', ls='--', lw=1.2, label='Target h/lambda = 0.1')
        ax.set_xlabel('Edge length / wavelength')
        ax.set_ylabel('CDF')
        ax.set_title('Airplane Mesh Electrical Resolution Diagnostic')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGS / 'heuristic_mesh_electrical_resolution.png', dpi=180)
    except Exception as e:
        print(f'[warn] Could not generate mesh electrical resolution figure: {e}')

print('Saved heuristic figures in', FIGS)
