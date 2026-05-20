"""Minimal TS05 usage example."""

import numpy as np

from ts05 import t04_field, t04_field_batch


def main():
    parmod = np.array(
        [2.0, -20.0, 0.0, -5.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        dtype=np.float64,
    )
    tilt = 0.1

    bx, by, bz = t04_field(parmod, tilt, -5.0, 1.0, 2.0)
    print(f"single point: B = ({bx:.3f}, {by:.3f}, {bz:.3f}) nT")

    xyz = np.array(
        [
            [-5.0, 1.0, 2.0],
            [-6.0, -1.0, 0.5],
        ],
        dtype=np.float64,
    )
    out = np.empty_like(xyz)
    t04_field_batch(parmod, tilt, xyz, out)
    print("batch field:")
    print(out)


if __name__ == "__main__":
    main()
