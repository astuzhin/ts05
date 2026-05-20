# ts05-numba

Small Python helpers for the Tsyganenko/Sitnov TS05 external magnetic field model.

The package exposes a Numba-callable API while preserving the original Fortran
TS05 calculation as the native backend. If the platform-specific shared library
is missing, it is built on first import with `gfortran`.

## Install

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e .
```

Requirements:

- Python 3.10+
- NumPy
- Numba
- `gfortran` available on `PATH`

## Basic Usage

```python
import numpy as np
from ts05 import t04_field

parmod = np.array([
    2.0,    # Pdyn
    -20.0,  # SYM-H / DST-like index
    0.0,    # IMF By GSM
    -5.0,   # IMF Bz GSM
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6,  # W1..W6
], dtype=np.float64)

tilt = 0.1
bx, by, bz = t04_field(parmod, tilt, -5.0, 1.0, 2.0)
```
