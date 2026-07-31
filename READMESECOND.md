# Student Surrogate Package

This folder is a small, zip-ready package for understanding the reduced
sender/receiver surrogate model and its relationship to the FiPy reference
model.

## Files In This Folder

- `genelet_sender_receiver_surrogate.py`
  - reduced-order time-domain surrogate
  - replaces the 2D diffusion PDE with a few transport compartments
  - uses backward Euler + Newton iterations for stable time stepping
- `genelet_sender_receiver_fipy.py`
  - reference Python PDE model
  - solves the full 2D reaction-diffusion system with FiPy
- `surrogate_vs_fipy_benchmark.json`
  - summary of one standard comparison case, including runtime and final outputs

## What The Surrogate Is

The full sender/receiver model is spatial: a sender hydrogel produces `S2`,
`S2` diffuses through the bath, and the receiver hydrogel consumes it through
binding to `I2` and `Th2`.

The surrogate keeps the receiver chemistry, but replaces the spatial PDE with
three well-mixed `S2` compartments:

1. sender gel
2. effective path / bath
3. receiver gel

The receiver chemistry is still:

- `S2 + I2 <-> S2:I2`
- `S2 + Th2 <-> S2:Th2`

So the surrogate preserves the main causal chain:

`sender production -> transport delay / attenuation -> receiver chemistry`

## Why This Reduction Works Here

This reduction is unusually natural for this system because:

- only `S2` diffuses
- the sender and receiver switches are immobilized
- the main outputs of interest are receiver-averaged time traces

That means the expensive part of the PDE is mostly representing transport of
one signal from sender to receiver through space. A small compartment model can
approximate that transport much more cheaply than a 2D mesh.

To better mimic the full 2D bath, the surrogate also includes a leakage term
from the path compartment. That term stands in for `S2` diffusing into the rest
of the bath instead of continuing directly to the receiver.

## Why The Surrogate Is More Stable

The FiPy model is already much more robust than the explicit NumPy model, but
it still solves a nonlinear PDE over many mesh cells and many time steps.

The surrogate solves only a 7-state nonlinear ODE system:

- `S_sender`
- `S_path`
- `S_receiver`
- `I2`
- `S2_I2`
- `Th2`
- `S2_Th2`

Each time step is solved with backward Euler and Newton iterations, so the
surrogate:

- does not have an explicit diffusion CFL limit
- remains stable at much larger time steps
- is much faster for time-domain sweeps

## When To Use It

Use the surrogate when you want:

- fast time-domain trends
- parameter sweeps over distance or threshold
- a teaching model for understanding transport + local chemistry
- a stable replacement for the explicit finite-difference prototype

Do not use the surrogate when you need:

- spatial concentration fields
- mesh-convergence studies
- transport-front detail at very early times
- final reference values for publication without cross-checking

## Suggested Reading Order

1. Read `genelet_sender_receiver_surrogate.py` first.
2. Then read `genelet_sender_receiver_fipy.py` to see what was reduced away.
3. Then open `surrogate_vs_fipy_benchmark.json` to see the accuracy/runtime tradeoff.

## Suggested First Runs

Surrogate:

```bash
python genelet_sender_receiver_surrogate.py \
  --hours 1.0 \
  --dt-s 300 \
  --distance-um 150 \
  --threshold-uM 5
```

FiPy:

```bash
python genelet_sender_receiver_fipy.py \
  --hours 1.0 \
  --dt-s 30 \
  --dx-um 25 \
  --distance-um 150 \
  --threshold-uM 5
```

## Benchmark Summary

On the standard 1-hour comparison case in this repo:

- center distance: `150 um`
- threshold: `5 uM`

The saved FiPy reference is about:

- receiver `I2`: `1.2104 nM`
- receiver total RNA: `5338.0 nM`

The surrogate gives about:

- receiver `I2`: `1.16 nM`
- receiver total RNA: `5349 nM`

So for this case, the surrogate reproduces the final receiver outputs fairly
well while being much faster.

From warm-run timings in the simulation conda environment:

- FiPy: about `31.75 s`
- surrogate, same `dt = 30 s`: about `0.98 s`
- surrogate, default `dt = 300 s`: about `0.93 s`

That puts the surrogate at roughly `30x` to `35x` faster on this benchmark.

## Practical Guidance For A Student

- Treat FiPy as the reference Python model.
- Treat the surrogate as a fast reduced model for sweeps and intuition.
- If a surrogate trend looks interesting, verify it with FiPy before drawing a
  strong conclusion.
- If you change the geometry or move far from the benchmark case, expect to
  retune or at least revalidate the surrogate.
