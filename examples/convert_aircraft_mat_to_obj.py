#!/usr/bin/env python3
"""Convert a POFacets-style .mat mesh (coord, facet) to OBJ.

Expected MAT keys:
  - coord: (Nv, 3) float vertex coordinates
  - facet: (Nt, >=3) triangle node indices in 1-based indexing

Usage:
  python convert_aircraft_mat_to_obj.py input.mat output.obj
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def _as_2d(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    return arr


def convert_mat_to_obj(input_mat: Path, output_obj: Path) -> None:
    data = loadmat(str(input_mat))
    if "coord" not in data or "facet" not in data:
        keys = sorted(k for k in data.keys() if not k.startswith("__"))
        raise KeyError(
            "MAT file must contain 'coord' and 'facet'. "
            f"Available keys: {keys}"
        )

    coord = _as_2d(np.asarray(data["coord"], dtype=float), "coord")
    facet = _as_2d(np.asarray(data["facet"]), "facet")

    if coord.shape[1] != 3:
        raise ValueError(f"coord must have shape (Nv,3), got {coord.shape}")
    if facet.shape[1] < 3:
        raise ValueError(f"facet must have at least 3 columns, got {facet.shape}")

    tri = np.asarray(facet[:, :3], dtype=np.int64)

    # Heuristic: POFacets stores 1-based indices. If we detect 0-based, shift.
    if tri.min() == 0:
        tri = tri + 1

    if tri.min() < 1 or tri.max() > coord.shape[0]:
        raise ValueError(
            "Triangle indices out of bounds after normalization: "
            f"min={tri.min()}, max={tri.max()}, Nv={coord.shape[0]}"
        )

    output_obj.parent.mkdir(parents=True, exist_ok=True)
    with output_obj.open("w", encoding="utf-8") as f:
        f.write(f"# Converted from {input_mat.name}\n")
        for x, y, z in coord:
            f.write(f"v {x:.12g} {y:.12g} {z:.12g}\n")
        for i, j, k in tri:
            f.write(f"f {int(i)} {int(j)} {int(k)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_mat", type=Path)
    parser.add_argument("output_obj", type=Path)
    args = parser.parse_args()
    convert_mat_to_obj(args.input_mat, args.output_obj)
