"""Numba-callable TS05 magnetic field bridge.

This module keeps the original TS05 Fortran implementation as the numerical
kernel and exposes scalar functions that can be called from Numba nopython
code. If the platform-specific shared library is missing, it is built with
gfortran on import.
"""

from __future__ import annotations

import ctypes
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
from numba import njit


def _library_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "libts05.dylib"
    if system == "Linux":
        return "libts05.so"
    if system == "Windows":
        return "ts05.dll"
    return "libts05.so"


def _build_library(lib_path: Path) -> None:
    root = Path(__file__).resolve().parent
    fixed_source = root / "ts05_fixed.f"
    shim_source = root / "ts05_shim.f90"

    if not fixed_source.exists() or not shim_source.exists():
        raise RuntimeError(
            "Cannot build TS05 native library: ts05_fixed.f and ts05_shim.f90 "
            f"must be next to {Path(__file__).name}"
        )

    if shutil.which("gfortran") is None:
        raise RuntimeError("Cannot build TS05 native library: gfortran was not found on PATH")

    cmd = [
        "gfortran",
        "-O3",
        "-fPIC",
        "-shared",
        str(fixed_source),
        str(shim_source),
        "-o",
        str(lib_path),
    ]
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Cannot build TS05 native library with gfortran.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


_LIB_PATH = Path(__file__).resolve().with_name(_library_name())
if not _LIB_PATH.exists():
    _build_library(_LIB_PATH)
_LIB = ctypes.CDLL(str(_LIB_PATH))

_t04_component = _LIB.t04_component
_t04_component.argtypes = (
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
)
_t04_component.restype = ctypes.c_double

_t04_fill = _LIB.t04_fill
_t04_fill.argtypes = (
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
)
_t04_fill.restype = None


@njit
def t04_field_scalar(
    p1: float,
    p2: float,
    p3: float,
    p4: float,
    p5: float,
    p6: float,
    p7: float,
    p8: float,
    p9: float,
    p10: float,
    ps: float,
    x: float,
    y: float,
    z: float,
) -> tuple[float, float, float]:
    """Return TS05 external field components in GSM coordinates, nT."""

    bx = _t04_component(0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, ps, x, y, z)
    by = _t04_component(1, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, ps, x, y, z)
    bz = _t04_component(2, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, ps, x, y, z)
    return bx, by, bz


@njit
def t04_field_scalar_into(
    p1: float,
    p2: float,
    p3: float,
    p4: float,
    p5: float,
    p6: float,
    p7: float,
    p8: float,
    p9: float,
    p10: float,
    ps: float,
    x: float,
    y: float,
    z: float,
    out: np.ndarray,
) -> None:
    """Fill `out[0:3]` with TS05 field components using one native call."""

    _t04_fill(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, ps, x, y, z, out.ctypes)


@njit
def t04_field_into(parmod: np.ndarray, ps: float, x: float, y: float, z: float, out: np.ndarray) -> None:
    """Fill `out[0:3]` with TS05 external field components in GSM coordinates, nT.

    This is the preferred function inside hot Numba integrators because it
    avoids temporary array allocation and evaluates the native TS05 kernel once.
    """

    t04_field_scalar_into(
        parmod[0],
        parmod[1],
        parmod[2],
        parmod[3],
        parmod[4],
        parmod[5],
        parmod[6],
        parmod[7],
        parmod[8],
        parmod[9],
        ps,
        x,
        y,
        z,
        out,
    )


@njit
def t04_field(parmod: np.ndarray, ps: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    """Return TS05 external field components in GSM coordinates, nT.

    `parmod` must be a one-dimensional float64 array with 10 values:
    PDYN, DST, BYIMF, BZIMF, W1, W2, W3, W4, W5, W6.
    """

    out = np.empty(3, dtype=np.float64)
    t04_field_into(parmod, ps, x, y, z, out)
    return out[0], out[1], out[2]


@njit
def t04_field_batch(parmod: np.ndarray, ps, xyz: np.ndarray, out: np.ndarray) -> None:
    """Fill `out[:, 0:3]` with TS05 field values for `xyz[:, 0:3]`.

    Supported input shapes:
    - `parmod.shape == (10,)` and scalar `ps`: one TS05 state for all points.
    - `parmod.shape == (n, 10)` and `ps.shape == (n,)`: one TS05 state per point.
    """

    if parmod.ndim == 1:
        for i in range(xyz.shape[0]):
            t04_field_into(parmod, ps, xyz[i, 0], xyz[i, 1], xyz[i, 2], out[i])
    else:
        for i in range(xyz.shape[0]):
            t04_field_into(parmod[i], ps[i], xyz[i, 0], xyz[i, 1], xyz[i, 2], out[i])
