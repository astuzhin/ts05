"""Example: compute TS05 magnetic field from OMNI/TS05 input files.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ts05 import data as ts05_data
from ts05 import t04_field, t04_field_batch

# Timed to compute the field
t = pd.Timestamp('2014-02-28', tz='UTC')

# Load solar wind and TS05 parameters at the given time
parmod, ps = ts05_data.inputs_at(ts05_data.load_omni_year(2014), t)

# Define the GSM grid
x_gsm = np.linspace(-30, 30, 100)
z_gsm = np.linspace(-15, 15, 100)
X, Z = np.meshgrid(x_gsm, z_gsm)
xyz = np.column_stack((X.ravel(), np.zeros_like(X).ravel(), Z.ravel()))

# Compute the field on the grid using parameters at the given time
# b is a 2D array of shape (200, 200), t04_field_batch computes the field and put it in b variable
b = np.empty_like(xyz)
t04_field_batch(parmod, ps, xyz, b)
b = np.linalg.norm(b, axis=1)

# Plot the field
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
fig, ax = plt.subplots(figsize=(10, 6))
cax = ax.pcolormesh(X, Z, b.reshape(X.shape), norm=LogNorm(vmin=1, vmax=500), cmap='jet')
fig.colorbar(cax, ax=ax, label='B [nT]')
ax.set_title(t)
ax.set_xlabel('X_GSM [R_E]')
ax.set_ylabel('Z_GSM [R_E]')
plt.show()
plt.savefig('map.png')
