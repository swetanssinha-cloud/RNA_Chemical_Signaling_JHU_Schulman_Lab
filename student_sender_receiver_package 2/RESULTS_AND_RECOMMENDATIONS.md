# Results And Recommendations

## What Was Tested

I compared the naive NumPy model and the FiPy model on the same coarse test case:

- sender/receiver center distance: `150 um`
- threshold in receiver: `5 uM`
- total simulated time: `1 hour`
- mesh spacing: `25 um`

Time stepping differed because the methods are different:

- FiPy: `dt = 30 s`
- NumPy: `dt = 0.5 s`

## Numerical Comparison

From `comparison_summary.json`:

- Final receiver `I2`:
  - FiPy: `1.2104 nM`
  - NumPy: `1.2038 nM`
  - relative difference: about `0.54%`
- Final receiver total RNA:
  - FiPy: `5338.0 nM`
  - NumPy: `5340.9 nM`
  - relative difference: about `0.055%`

## Interpretation

For this 2-node problem, the naive NumPy model is a reasonable starting point.

That does not mean it should replace FiPy or COMSOL. It means:

- it is good enough to help a student understand the geometry, masks, diffusion, and reaction terms
- it appears to reproduce the coarse FiPy dynamics well for at least this test case
- it is suitable for quick debugging and qualitative intuition

The main risk is that the NumPy model can quietly become unreliable when:

- the mesh gets finer
- the explicit time step is too large
- the kinetics become stiffer
- the model is extended to larger or more complex networks

## Recommendation

Use the tools in this order:

1. `genelet_sender_receiver_numpy.py`
   - First pass only.
   - Use it to understand the equations and to make simple edits safely.
2. `genelet_sender_receiver_fipy.py`
   - Main Python model.
   - Use this for sweeps, comparisons, and anything that should be trusted numerically.
3. `genelet_sender_receiver_mph.py`
   - Use this when you want to compare directly against the real COMSOL model.
   - Treat COMSOL as the reference implementation when model details are ambiguous.
4. `genelet_sender_receiver_mph_builder.py`
   - Use this when you want a fresh COMSOL model generated from Python without relying on the paper `.mph` file.
   - Current status: build/save/short solve works, and short-time agreement is now reasonable.

## COMSOL Note

The COMSOL helper script is included and should be a practical starting point, but it was not fully verified end-to-end here.

What was verified:

- COMSOL executable was found at `/Applications/COMSOL64/Multiphysics/bin/macarm64/comsol`
- `COMSOL Multiphysics 6.4.0.343` reported correctly
- the Python `mph` package was installed and importable

What was not fully verified:

- loading and listing the full `COMSOL-Model-SI-Chap-2-4.mph` file to completion from Python on this machine

What was verified for the fresh COMSOL builder:

- COMSOL can be launched from Python `mph`
- a fresh sender/receiver model can be built and saved from Python
- a very short transient solve now completes successfully after correcting the bimolecular rate units
- after correcting geometry localization and diffusion scaling, the short-time COMSOL result is now in the same ballpark as the FiPy and NumPy results

What was not yet verified for the fresh COMSOL builder:

- longer-time agreement with the FiPy and NumPy models
- robustness across broader parameter sweeps

So the student should use:

```bash
python genelet_sender_receiver_mph.py --mph-file /path/to/model.mph --list-only
```

as the first COMSOL-side test before trying parameter overrides or solves.

## Additional Short-Time Check

I also reran the FiPy and NumPy models on the same very short case used for the COMSOL smoke solve:

- `distance = 150 um`
- `threshold = 5 uM`
- `time = 0.001 h = 3.6 s`
- `dx = 25 um`

Short-time results:

- FiPy final receiver `I2`: `100.0 nM`
- NumPy final receiver `I2`: `99.999996 nM`
- COMSOL final receiver `I2`: `99.999997 nM`
- FiPy final receiver total RNA: `0.00392 nM`
- NumPy final receiver total RNA: `0.00235 nM`
- COMSOL final receiver total RNA: `0.00177 nM`

This means all three models are now in reasonable agreement at short times.

So the fresh COMSOL builder is now:

- structurally working
- able to solve
- short-time cross-validated to a reasonable first approximation
- not yet validated for longer times or broader sweeps

That is now a much stronger starting point for a student, but it should still be presented as a new reconstruction rather than a finished replacement for the paper COMSOL model.

## Real Section 2.1 Preset Check

After the real `COMSOL-Model-SI-Chap-2-1.mph` file was downloaded locally, I extracted its main geometry and parameter values and added a `comsol-2-1` preset to the Python models.

That preset uses:

- `node length = 75 um`
- `center distance = 175 um`
- `bath margin = 2375 um`
- `Dgel = 42 um^2/s`
- `Dsolution = 150 um^2/s`
- `sender switch = 100 nM`
- `receiver switch = 100 nM`
- `threshold = 10 uM`

I then ran the same short `0.001 h` case with that preset.

Results:

- NumPy `dx = 50 um`, `dt = 0.1 s`
  - receiver `I2 = 99.999996 nM`
  - receiver total RNA `= 0.00408 nM`
- FiPy `dx = 50 um`, `dt = 0.2 s`
  - receiver `I2 = 100.0 nM`
  - receiver total RNA `= 0.00613 nM`
- FiPy `dx = 25 um`, `dt = 0.1 s`
  - receiver `I2 = 100.0 nM`
  - receiver total RNA `= 0.00266 nM`
- fresh COMSOL builder, mesh level `8`
  - receiver `I2 = 99.999990 nM`
  - receiver total RNA `= 0.01048 nM`

Interpretation:

- Yes, FiPy can run the real `2.1`-aligned version.
- Receiver `I2` is effectively identical across all of these short runs.
- Receiver total RNA is still more sensitive to the discretization and implementation details.

So the correct conclusion is:

- the Python models can reproduce the real `2.1` setup structure and run it
- the exact `2.1` short case is qualitatively consistent across implementations
- but the total-RNA observable is not yet tight enough to call fully cross-validated

For the student, that means:

- use the `comsol-2-1` preset as a strong starting point
- use FiPy as the main Python reference
- use the original paper COMSOL model as the final authority if exact quantitative agreement matters

## Short Convergence Checks

### FiPy

I ran a short-time 3-level FiPy refinement test with:

- `dx = 25, 16.67, 11.11 um`
- matched time steps `dt = 0.1, 0.0667, 0.0444 s`

Results:

- receiver `I2` stayed at `100 nM` on all three levels
- receiver total RNA decreased with refinement:
  - `0.00392 nM`
  - `0.00239 nM`
  - `0.00164 nM`

The Richardson estimate for receiver total RNA gave:

- observed order `p ≈ 1.77`

That is a good sign. It suggests the FiPy solution is behaving smoothly enough for a formal convergence analysis on this short case.

### COMSOL

I ran a short-time COMSOL mesh-sensitivity check on mesh levels `6`, `7`, and `8`.

Results:

- receiver `I2` was identical to the shown precision across all three levels
- receiver total RNA was also identical to the shown precision across all three levels

This is encouraging, but it is not a formal Richardson test because COMSOL's built-in mesh levels are not a strict fixed-ratio mesh family.

## Recommendation After These Checks

- Use FiPy as the main reference implementation for mesh-convergence work.
- Treat the COMSOL mesh-level sweep as a stability check, not a formal order-of-accuracy estimate.
- If you want a formal COMSOL Richardson study later, use explicitly controlled element sizes rather than mesh levels.

## Practical Advice For The Student

- Do not change both geometry and kinetics at the same time in the first round.
- First verify that the NumPy and FiPy scripts produce sensible trends when distance changes.
- Keep a coarse grid while learning the model.
- Only tighten the mesh after the trends make sense.
- When comparing against COMSOL, compare one observable at a time:
  - receiver `I2`
  - receiver total RNA
- Save every run with parameter values in the file name or in a small JSON sidecar.
- If working in COMSOL first, get the 2.1 sender/receiver `.mph` file locally before tuning the builder script further.
