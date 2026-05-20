"""Small helpers for OMNI files with precomputed TS05 variables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit


HEADER = (
    "IYEAR",
    "IDAY",
    "IHOUR",
    "MIN",
    "BXGSM",
    "BYGSM",
    "BZGSM",
    "VXGSE",
    "VYGSE",
    "VZGSE",
    "DEN",
    "TEMP",
    "SYMH",
    "IMFFLAG",
    "ISWFLAG",
    "TILT",
    "Pdyn",
    "W1",
    "W2",
    "W3",
    "W4",
    "W5",
    "W6",
)

PARMOD_COLUMNS = (16, 12, 5, 6, 17, 18, 19, 20, 21, 22)


def datetime_to_minute(when) -> int:
    """Convert a datetime-like value to integer minutes since Unix epoch."""

    if isinstance(when, (int, np.integer)):
        return int(when)
    if isinstance(when, (float, np.floating)):
        return int(np.floor(float(when) / 60.0))
    if hasattr(when, "to_datetime64"):
        return int(np.datetime64(when.to_datetime64(), "m").astype("int64"))
    return int(np.datetime64(when, "m").astype("int64"))


def times_to_minutes(when) -> np.ndarray:
    """Convert array-like times to integer minutes since Unix epoch.

    Numeric arrays are interpreted as UTC seconds. Datetime-like arrays,
    including pandas DatetimeIndex, are interpreted as absolute datetimes.
    """

    if hasattr(when, "to_numpy"):
        values = when.to_numpy()
    else:
        values = np.asarray(when)

    if values.ndim == 0:
        return np.array([datetime_to_minute(when)], dtype=np.int64)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[m]").astype(np.int64)
    if np.issubdtype(values.dtype, np.number):
        return np.floor(values.astype(np.float64) / 60.0).astype(np.int64)
    return np.array([datetime_to_minute(t) for t in values], dtype=np.int64)


def load_omni_year(year: int, data_dir: str | Path = "data", quality: str = "all") -> dict:
    """Load one yearly OMNI/TS05 file.

    `quality="good"` keeps only rows with IMFFLAG == 1 and ISWFLAG == 1.
    The returned `parmod` columns are ordered for TS05:
    Pdyn, SYMH, BYGSM, BZGSM, W1, W2, W3, W4, W5, W6.
    """

    path = Path(data_dir) / f"{int(year)}_OMNI_5m_with_TS05_variables.dat"
    raw = np.loadtxt(path, skiprows=1, dtype=np.float64)
    raw = np.atleast_2d(raw)

    imf_flag = raw[:, 13].astype(np.int64)
    sw_flag = raw[:, 14].astype(np.int64)
    if quality == "good":
        mask = (imf_flag == 1) & (sw_flag == 1)
        raw = raw[mask]
        imf_flag = imf_flag[mask]
        sw_flag = sw_flag[mask]
    elif quality != "all":
        raise ValueError("quality must be 'all' or 'good'")

    years = raw[:, 0].astype(np.int64)
    doy = raw[:, 1].astype(np.int64)
    hour = raw[:, 2].astype(np.int64)
    minute = raw[:, 3].astype(np.int64)

    time_min = np.empty(raw.shape[0], dtype=np.int64)
    for y in np.unique(years):
        base = datetime_to_minute(f"{int(y):04d}-01-01T00:00")
        rows = years == y
        time_min[rows] = base + (doy[rows] - 1) * 1440 + hour[rows] * 60 + minute[rows]

    return {
        "time_min": time_min,
        "parmod": np.ascontiguousarray(raw[:, PARMOD_COLUMNS], dtype=np.float64),
        "tilt": np.ascontiguousarray(raw[:, 15], dtype=np.float64),
        "imf_flag": imf_flag,
        "sw_flag": sw_flag,
        "raw": raw,
        "columns": HEADER,
        "path": str(path),
    }


def load_omni_range(start_year: int, stop_year: int, data_dir: str | Path = "data", quality: str = "all") -> dict:
    """Load and concatenate several yearly files, inclusive."""

    parts = [load_omni_year(year, data_dir=data_dir, quality=quality) for year in range(int(start_year), int(stop_year) + 1)]
    if not parts:
        raise ValueError("empty year range")

    return {
        "time_min": np.concatenate([p["time_min"] for p in parts]),
        "parmod": np.ascontiguousarray(np.vstack([p["parmod"] for p in parts]), dtype=np.float64),
        "tilt": np.ascontiguousarray(np.concatenate([p["tilt"] for p in parts]), dtype=np.float64),
        "imf_flag": np.concatenate([p["imf_flag"] for p in parts]),
        "sw_flag": np.concatenate([p["sw_flag"] for p in parts]),
        "columns": HEADER,
        "paths": [p["path"] for p in parts],
    }


def inputs_at(data: dict, when, mode: str = "nearest", max_gap_minutes: int | None = 30):
    """Return `(parmod, tilt)` for one or many times.

    For scalar `when`, returns `parmod.shape == (10,)` and scalar `tilt`.
    For array-like `when`, returns `parmod.shape == (n, 10)` and `tilt.shape == (n,)`.
    Lookup mode is one of `nearest`, `previous`, or `linear`.
    """

    time_min = data["time_min"]
    parmods = data["parmod"]
    tilts = data["tilt"]
    max_gap = -1 if max_gap_minutes is None else int(max_gap_minutes)

    if mode not in ("nearest", "previous", "linear"):
        raise ValueError("mode must be 'nearest', 'previous', or 'linear'")

    if np.ndim(when) == 0:
        t_min = datetime_to_minute(when)
        out = np.empty(10, dtype=np.float64)
        if mode == "nearest":
            ps, ok = nearest_inputs_at_minute(t_min, time_min, parmods, tilts, out, max_gap)
        elif mode == "previous":
            ps, ok = previous_inputs_at_minute(t_min, time_min, parmods, tilts, out, max_gap)
        else:
            ps, ok = linear_inputs_at_minute(t_min, time_min, parmods, tilts, out, max_gap)

        if not ok:
            raise ValueError("no OMNI/TS05 input is available within max_gap_minutes")
        return out, ps

    t_min = times_to_minutes(when)
    out = np.empty((t_min.shape[0], 10), dtype=np.float64)
    ps = np.empty(t_min.shape[0], dtype=np.float64)
    ok = np.empty(t_min.shape[0], dtype=np.bool_)

    if mode == "nearest":
        nearest_inputs_at_minutes(t_min, time_min, parmods, tilts, out, ps, ok, max_gap)
    elif mode == "previous":
        previous_inputs_at_minutes(t_min, time_min, parmods, tilts, out, ps, ok, max_gap)
    else:
        linear_inputs_at_minutes(t_min, time_min, parmods, tilts, out, ps, ok, max_gap)

    if not np.all(ok):
        out[~ok, :] = np.nan
        ps[~ok] = np.nan
    return out, ps


def to_numba_arrays(data: dict):
    """Return the arrays needed inside Numba code: time_min, parmod, tilt."""

    return data["time_min"], data["parmod"], data["tilt"]


@njit
def _lower_bound(values, target):
    lo = 0
    hi = values.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit
def _copy_parmod(src, row, out):
    for j in range(10):
        out[j] = src[row, j]


@njit
def nearest_inputs_at_minute(t_min, time_min, parmods, tilts, parmod_out, max_gap_minutes=30):
    """Fill `parmod_out` from nearest record. Returns `(tilt, valid)`."""

    n = time_min.shape[0]
    if n == 0:
        return 0.0, False

    i = _lower_bound(time_min, t_min)
    if i == 0:
        idx = 0
    elif i == n:
        idx = n - 1
    else:
        left_gap = t_min - time_min[i - 1]
        right_gap = time_min[i] - t_min
        if left_gap <= right_gap:
            idx = i - 1
        else:
            idx = i

    gap = time_min[idx] - t_min
    if gap < 0:
        gap = -gap
    if max_gap_minutes >= 0 and gap > max_gap_minutes:
        return 0.0, False

    _copy_parmod(parmods, idx, parmod_out)
    return tilts[idx], True


@njit
def nearest_inputs_at_minutes(t_mins, time_min, parmods, tilts, parmod_out, tilt_out, valid_out, max_gap_minutes=30):
    """Batch version of `nearest_inputs_at_minute`."""

    for i in range(t_mins.shape[0]):
        ps, ok = nearest_inputs_at_minute(t_mins[i], time_min, parmods, tilts, parmod_out[i], max_gap_minutes)
        tilt_out[i] = ps
        valid_out[i] = ok


@njit
def previous_inputs_at_minute(t_min, time_min, parmods, tilts, parmod_out, max_gap_minutes=30):
    """Fill `parmod_out` from last record not later than `t_min`. Returns `(tilt, valid)`."""

    i = _lower_bound(time_min, t_min + 1) - 1
    if i < 0:
        return 0.0, False

    gap = t_min - time_min[i]
    if max_gap_minutes >= 0 and gap > max_gap_minutes:
        return 0.0, False

    _copy_parmod(parmods, i, parmod_out)
    return tilts[i], True


@njit
def previous_inputs_at_minutes(t_mins, time_min, parmods, tilts, parmod_out, tilt_out, valid_out, max_gap_minutes=30):
    """Batch version of `previous_inputs_at_minute`."""

    for i in range(t_mins.shape[0]):
        ps, ok = previous_inputs_at_minute(t_mins[i], time_min, parmods, tilts, parmod_out[i], max_gap_minutes)
        tilt_out[i] = ps
        valid_out[i] = ok


@njit
def linear_inputs_at_minute(t_min, time_min, parmods, tilts, parmod_out, max_gap_minutes=30):
    """Linearly interpolate TS05 inputs. Returns `(tilt, valid)`."""

    n = time_min.shape[0]
    if n == 0:
        return 0.0, False

    i = _lower_bound(time_min, t_min)
    if i < n and time_min[i] == t_min:
        _copy_parmod(parmods, i, parmod_out)
        return tilts[i], True
    if i == 0 or i == n:
        return 0.0, False

    t0 = time_min[i - 1]
    t1 = time_min[i]
    span = t1 - t0
    if span <= 0:
        return 0.0, False
    if max_gap_minutes >= 0 and span > max_gap_minutes:
        return 0.0, False

    a = (t_min - t0) / span
    for j in range(10):
        parmod_out[j] = parmods[i - 1, j] * (1.0 - a) + parmods[i, j] * a
    return tilts[i - 1] * (1.0 - a) + tilts[i] * a, True


@njit
def linear_inputs_at_minutes(t_mins, time_min, parmods, tilts, parmod_out, tilt_out, valid_out, max_gap_minutes=30):
    """Batch version of `linear_inputs_at_minute`."""

    for i in range(t_mins.shape[0]):
        ps, ok = linear_inputs_at_minute(t_mins[i], time_min, parmods, tilts, parmod_out[i], max_gap_minutes)
        tilt_out[i] = ps
        valid_out[i] = ok
