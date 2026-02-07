#!/usr/bin/env python3
"""Run a Bempp-cl impedance-loaded far-field reference for cross-validation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Tuple

import numpy as np


def ensure_gmsh_on_path() -> None:
    venv_bin = str(Path(sys.executable).parent)
    path = os.environ.get("PATH", "")
    entries = path.split(os.pathsep) if path else []
    if venv_bin not in entries:
        os.environ["PATH"] = venv_bin + os.pathsep + path


ensure_gmsh_on_path()


try:
    import bempp_cl.api as bempp
    from bempp_cl.api.linalg import lu
except ImportError as exc:
    raise SystemExit(
        "Bempp-cl is required. Install it first, then rerun this script.\n"
        "Example: pip install bempp-cl"
    ) from exc


C0 = 299_792_458.0


def spherical_sampling_grid(
    n_theta: int, n_phi: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dtheta = np.pi / n_theta
    dphi = 2.0 * np.pi / n_phi
    theta = (np.arange(n_theta) + 0.5) * dtheta
    phi = (np.arange(n_phi) + 0.5) * dphi
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    theta_flat = theta_grid.ravel()
    phi_flat = phi_grid.ravel()
    directions = np.vstack(
        (
            np.sin(theta_flat) * np.cos(phi_flat),
            np.sin(theta_flat) * np.sin(phi_flat),
            np.cos(theta_flat),
        )
    )
    weights = np.sin(theta_flat) * dtheta * dphi
    return theta_flat, phi_flat, directions, weights, np.array([dtheta, dphi])


def run_bempp_impedance(
    freq_hz: float,
    aperture_lambda: float,
    mesh_step_lambda: float,
    zs_imag_ohm: float,
    theta_inc_deg: float,
    phi_inc_deg: float,
    n_theta: int,
    n_phi: int,
    op_sign: str,
    rhs_cross: str,
    rhs_sign: float,
    phase_sign: str,
    zs_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    lambda0 = C0 / freq_hz
    k0 = 2.0 * np.pi / lambda0
    side = aperture_lambda * lambda0
    mesh_h = mesh_step_lambda * lambda0

    corners = np.array(
        [
            [-side / 2.0, -side / 2.0, 0.0],
            [side / 2.0, -side / 2.0, 0.0],
            [side / 2.0, side / 2.0, 0.0],
            [-side / 2.0, side / 2.0, 0.0],
        ]
    )
    grid = bempp.shapes.screen(corners=corners, h=mesh_h)
    rwg = bempp.function_space(grid, "RWG", 0)
    snc = bempp.function_space(grid, "SNC", 0)

    electric = bempp.operators.boundary.maxwell.electric_field(rwg, rwg, snc, k0)
    identity = bempp.operators.boundary.sparse.identity(rwg, rwg, snc)
    zs = 1j * zs_imag_ohm * zs_scale
    if op_sign == "minus":
        op = electric - zs * identity
    elif op_sign == "plus":
        op = electric + zs * identity
    else:
        raise ValueError(f"Unknown op_sign: {op_sign}")

    theta_inc = np.deg2rad(theta_inc_deg)
    phi_inc = np.deg2rad(phi_inc_deg)

    direction = np.array(
        [
            np.sin(theta_inc) * np.cos(phi_inc),
            np.sin(theta_inc) * np.sin(phi_inc),
            -np.cos(theta_inc),
        ],
        dtype=float,
    )
    # Theta polarization unit vector, perpendicular to propagation direction.
    polarization = np.array(
        [
            np.cos(theta_inc) * np.cos(phi_inc),
            np.cos(theta_inc) * np.sin(phi_inc),
            np.sin(theta_inc),
        ],
        dtype=np.complex128,
    )

    @bempp.complex_callable
    def tangential_trace(x, n, domain_index, result):
        if phase_sign == "plus":
            phase = np.exp(1j * k0 * np.dot(direction, x))
        elif phase_sign == "minus":
            phase = np.exp(-1j * k0 * np.dot(direction, x))
        else:
            raise ValueError(f"Unknown phase_sign: {phase_sign}")
        e_inc = polarization * phase
        if rhs_cross == "e_cross_n":
            trace = np.cross(e_inc, n)
        elif rhs_cross == "n_cross_e":
            trace = np.cross(n, e_inc)
        else:
            raise ValueError(f"Unknown rhs_cross: {rhs_cross}")
        result[:] = rhs_sign * trace

    rhs = bempp.GridFunction(rwg, fun=tangential_trace, dual_space=snc)
    surface_current = lu(op, rhs)

    theta_flat, phi_flat, directions, weights, steps = spherical_sampling_grid(
        n_theta=n_theta, n_phi=n_phi
    )
    far_field = bempp.operators.far_field.maxwell.electric_field(rwg, directions, k0)
    e_ff = -(far_field * surface_current)

    power = np.real(np.sum(np.conjugate(e_ff) * e_ff, axis=0))
    power = np.maximum(power, 1e-30)
    total = float(np.sum(power * weights))
    directivity = 4.0 * np.pi * power / total
    dir_dbi = 10.0 * np.log10(np.maximum(directivity, 1e-30))

    metadata = {
        "frequency_hz": freq_hz,
        "lambda_m": lambda0,
        "wavenumber_rad_per_m": k0,
        "aperture_lambda": aperture_lambda,
        "aperture_side_m": side,
        "mesh_step_lambda": mesh_step_lambda,
        "mesh_step_m": mesh_h,
        "zs_imag_ohm": zs_imag_ohm,
        "zs_scale": zs_scale,
        "zs_effective_imag_ohm": zs_imag_ohm * zs_scale,
        "theta_inc_deg": theta_inc_deg,
        "phi_inc_deg": phi_inc_deg,
        "op_sign": op_sign,
        "rhs_cross": rhs_cross,
        "rhs_sign": rhs_sign,
        "phase_sign": phase_sign,
        "n_theta": n_theta,
        "n_phi": n_phi,
        "dtheta_deg": float(np.rad2deg(steps[0])),
        "dphi_deg": float(np.rad2deg(steps[1])),
        "num_points": int(theta_flat.size),
        "num_vertices": int(grid.number_of_vertices),
        "num_elements": int(grid.number_of_elements),
    }
    return theta_flat, phi_flat, dir_dbi, metadata


def write_farfield_csv(path: Path, theta: np.ndarray, phi: np.ndarray, dir_dbi: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["theta_deg", "phi_deg", "dir_bempp_imp_dBi"])
        for t, p, d in zip(theta, phi, dir_dbi):
            writer.writerow([float(np.rad2deg(t)), float(np.rad2deg(p)), float(d)])


def write_phi0_cut_csv(
    path: Path, theta: np.ndarray, phi: np.ndarray, dir_dbi: np.ndarray, n_phi: int
) -> None:
    half_bin = 0.5 * (2.0 * np.pi / n_phi)
    phi_dist = np.minimum(phi, 2.0 * np.pi - phi)
    indices = np.where(phi_dist <= half_bin + 1e-12)[0]
    order = indices[np.argsort(theta[indices])]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["theta_deg", "dir_bempp_imp_dBi"])
        for i in order:
            writer.writerow([float(np.rad2deg(theta[i])), float(dir_dbi[i])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freq-ghz", type=float, default=3.0)
    parser.add_argument("--aperture-lambda", type=float, default=4.0)
    parser.add_argument("--mesh-step-lambda", type=float, default=1.0 / 3.0)
    parser.add_argument("--zs-imag-ohm", type=float, default=200.0)
    parser.add_argument("--theta-inc-deg", type=float, default=0.0)
    parser.add_argument("--phi-inc-deg", type=float, default=0.0)
    parser.add_argument("--n-theta", type=int, default=180)
    parser.add_argument("--n-phi", type=int, default=72)
    parser.add_argument("--op-sign", choices=["minus", "plus"], default="minus")
    parser.add_argument("--rhs-cross", choices=["e_cross_n", "n_cross_e"], default="e_cross_n")
    parser.add_argument("--rhs-sign", type=float, default=1.0)
    parser.add_argument("--phase-sign", choices=["plus", "minus"], default="plus")
    parser.add_argument("--zs-scale", type=float, default=1.0)
    parser.add_argument("--output-prefix", type=str, default="impedance")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root containing data/.",
    )
    args = parser.parse_args()

    data_dir = args.project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    theta, phi, dir_dbi, meta = run_bempp_impedance(
        freq_hz=args.freq_ghz * 1e9,
        aperture_lambda=args.aperture_lambda,
        mesh_step_lambda=args.mesh_step_lambda,
        zs_imag_ohm=args.zs_imag_ohm,
        theta_inc_deg=args.theta_inc_deg,
        phi_inc_deg=args.phi_inc_deg,
        n_theta=args.n_theta,
        n_phi=args.n_phi,
        op_sign=args.op_sign,
        rhs_cross=args.rhs_cross,
        rhs_sign=args.rhs_sign,
        phase_sign=args.phase_sign,
        zs_scale=args.zs_scale,
    )

    farfield_csv = data_dir / f"bempp_{args.output_prefix}_farfield.csv"
    cut_csv = data_dir / f"bempp_{args.output_prefix}_cut_phi0.csv"
    meta_json = data_dir / f"bempp_{args.output_prefix}_metadata.json"

    write_farfield_csv(farfield_csv, theta, phi, dir_dbi)
    write_phi0_cut_csv(cut_csv, theta, phi, dir_dbi, args.n_phi)
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved {farfield_csv}")
    print(f"Saved {cut_csv}")
    print(f"Saved {meta_json}")
    print(
        "Note: compare against Julia with "
        "python validation/bempp/compare_impedance_to_julia.py"
    )


if __name__ == "__main__":
    main()
