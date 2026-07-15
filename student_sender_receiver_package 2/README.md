# Sender/Receiver Modeling Package

This folder contains three versions of the same 2-node sender/receiver reaction-diffusion model from SI Chapter 2:

- `genelet_sender_receiver_numpy.py`
  - simplest version
  - explicit finite differences
  - best for learning the model structure
- `genelet_sender_receiver_fipy.py`
  - more numerically reliable Python version
  - recommended for parameter sweeps and actual exploratory work
- `genelet_sender_receiver_mph.py`
  - helper script for loading and running the existing COMSOL `.mph` model through Python `mph`
  - intended to work with the real COMSOL model file rather than rebuild the model from scratch
- `genelet_sender_receiver_mph_builder.py`
  - builds a fresh COMSOL sender/receiver model from scratch via Python `mph`
  - does not require the original paper `.mph` file

Also included:

- `comparison_summary.json`
  - direct numerical comparison of the FiPy and NumPy versions on the same coarse 1-hour test case
- `comparison_summary_short.json`
  - direct numerical comparison of the FiPy and NumPy versions on the same very short test case used for the COMSOL smoke solve
- `comparison_summary_short_with_comsol.json`
  - direct numerical comparison of FiPy, NumPy, and the corrected fresh COMSOL builder on the same very short test case
- `comparison_summary_comsol_2_1_short.json`
  - short comparison for the real `2.1` parameter preset across NumPy, FiPy, and the fresh COMSOL builder
- `fipy_convergence_short.json`
  - short-time FiPy mesh-refinement and Richardson-extrapolation summary
- `comsol_mesh_sensitivity_short.json`
  - short-time COMSOL mesh-sensitivity summary on mesh levels 6, 7, and 8
- `comsol_2_1_summary.json`
  - top-level parameter and geometry summary extracted from the real paper `COMSOL-Model-SI-Chap-2-1.mph`
- `fipy_sender_receiver_kinetics.png`
- `fipy_sender_receiver_fields.png`
- `numpy_sender_receiver_kinetics.png`
- `numpy_sender_receiver_fields.png`
- `numpy_sender_receiver_kinetics.csv`
- `numpy_sender_receiver_final_fields.npz`
- `test_builder_domainode_compose_short_solve_v3.mph`
  - corrected COMSOL model generated from Python and solved successfully for a very short test case
- `RESULTS_AND_RECOMMENDATIONS.md`

## Recommended Order

1. Start with `genelet_sender_receiver_numpy.py`.
2. Once the equations and geometry make sense, switch to `genelet_sender_receiver_fipy.py`.
3. Use `genelet_sender_receiver_mph.py` only when you want to inspect or rerun the existing COMSOL model.
4. Use `genelet_sender_receiver_mph_builder.py` if you want a Python-created COMSOL version of the same 2-node equations.

## Suggested First Runs

NumPy:

```bash
python genelet_sender_receiver_numpy.py \
  --hours 1.0 --dt-s 0.5 --dx-um 25 --distance-um 150 --threshold-uM 5
```

FiPy:

```bash
python genelet_sender_receiver_fipy.py \
  --hours 1.0 --dt-s 30 --dx-um 25 --distance-um 150 --threshold-uM 5
```

FiPy with the real `2.1` parameter preset:

```bash
python genelet_sender_receiver_fipy.py \
  --preset comsol-2-1 --hours 0.001 --dt-s 0.2 --dx-um 50
```

NumPy with the real `2.1` parameter preset:

```bash
python genelet_sender_receiver_numpy.py \
  --preset comsol-2-1 --hours 0.001 --dt-s 0.1 --dx-um 50
```

COMSOL/MPh:

```bash
python genelet_sender_receiver_mph.py \
  --mph-file /path/to/COMSOL-Model-SI-Chap-2-4.mph \
  --list-only
```

Then, after identifying the real COMSOL parameter names:

```bash
python genelet_sender_receiver_mph.py \
  --mph-file /path/to/COMSOL-Model-SI-Chap-2-4.mph \
  --set x=300[um] \
  --set L=50[um]
```

Fresh COMSOL builder:

```bash
python genelet_sender_receiver_mph_builder.py \
  --hours 0.001 --output-dt-s 1
```

## Environment Notes

- The FiPy and `mph` scripts were developed against a conda environment that had `numpy`, `matplotlib`, `fipy`, and `mph`.
- COMSOL was expected at:
  - `/Applications/COMSOL64/Multiphysics`
- The `mph` script prepends the COMSOL binary directory to `PATH` automatically.

## SI Constants Used

These are the constants used in the Python models and in the COMSOL builder:

- `Dgel = 60 um^2/s`
- `Dsolution = 150 um^2/s`
- `kp = 0.2 1/s`
- `kd_ds = 3e-4 1/s`
- `kd_ss = 3e-4 1/s`
- `kslow = 1e5 1/M/s`
- `kfast = 1e6 1/M/s`

These match the values listed in SI Chapter 2, Table 1.

For the real paper `2.1` preset, the scripts also support:

- `node_length = 75 um`
- `center_distance = 175 um`
- `bath_margin = 2375 um`
- `Dgel = 42 um^2/s`
- `Dsolution = 150 um^2/s`
- `sender switch = 100 nM`
- `receiver switch = 100 nM`
- `threshold = 10 uM`

These values were taken from the downloaded `COMSOL-Model-SI-Chap-2-1.mph` summary in `comsol_2_1_summary.json`.

## Current Status Summary

- NumPy and FiPy already used the correct SI constants.
- The fresh COMSOL builder now builds and completes a short solve.
- On the short `0.001 h` test case, the corrected COMSOL builder is now in reasonable agreement with FiPy and NumPy.
- The real paper `2.1` parameter preset now runs in both FiPy and NumPy.
- For that `2.1` preset, the three implementations are qualitatively consistent at short times but not yet fully cross-validated quantitatively.
- The fresh COMSOL builder is still under validation for longer times and broader parameter sweeps.
- A short FiPy convergence study and a short COMSOL mesh-sensitivity study are now included in the package.

## Important Numerical Note

The NumPy model is explicit Euler. That makes it easy to read, but it is not unconditionally stable. If you reduce `dx_um`, you must also reduce `dt_s`. The script checks a simple diffusion stability bound and will stop if `dt_s` is too large.
