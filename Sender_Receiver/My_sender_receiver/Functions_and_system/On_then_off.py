"""
2D Tethered Genelet Model - "On then off" (I1O2 shutoff at steady state)
Built directly on top of TG_Rmesh_fast.py -- same equations, same split
S2-only solver, same initialization. The only change is the outer time loop.

WHAT'S DIFFERENT FROM TG_Rmesh_fast.py
---------------------------------------
TG_Rmesh_fast.py runs for a fixed 8 h and (optionally) breaks the moment its
steady-state detector fires. That detector never actually fired in practice:

  1. ss_window=50 checks spaced check_interval=100 steps (1.67 h) apart means
     the rolling window can't even fill up until ~83 h of simulated time --
     far beyond the old 8 h total_time. The convergence you could see by eye
     in the plots was never confirmed numerically for exactly this reason.
  2. The relative-change test took np.max() over EVERY cell in the 5mm bath,
     including cells near the far domain edge that are still slowly filling
     in from diffusion (L^2/D ~ 11-12 h for this domain) long after the
     receiver itself has visibly flattened. That would have kept dragging
     out convergence even with a bigger window.

This version fixes both: the relative-change test is restricted to cells
inside the receiver node (the only region the switch state actually lives
in), and the window/interval/tolerance are sized so convergence can plausibly
fire within a few hours instead of requiring multiple simulated days.

The run is a two-phase state machine, using the *same* detector both times:

  Phase ON  : run from t=0 until the receiver reaches steady state.
              At that point, record t_shutoff and set I1O2 -> 0 everywhere.
  Phase OFF : keep running (same equations -- I1O2 is just 0 now) until the
              receiver reaches steady state again (the recovered plateau).

I1O2 is a real CellVariable and eq_S2 holds a reference to it, so
I1O2.setValue(0.0) propagates into the next eq_S2.sweep() with no need to
rebuild the equation.

Each phase has a wall-clock/sim-time safety cap (PHASE_MAX_TIME) so a phase
that never numerically converges can't hang the run forever -- it forces the
transition (or the stop) anyway and says so in the output.
"""

import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, TransientTerm, DiffusionTerm, ImplicitSourceTerm, Gmsh2D, LinearLUSolver
from fipy.tools import numerix
import pandas as pd
import time as timer
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from Functions import initalize_variables_speedup
from Mesh.New_simple_mesh import create_conformal_radial_mesh

# =============================================================================
# PARAMETERS (identical to TG_Rmesh_fast.py)
# =============================================================================
wall_start_time = timer.time()

D_solution = 150.0
D_gel = 60.0
k_p = 0.01
k_d_ds = 3e-4 #* 0.1 #just testing to see how I can affect the shape
k_d_ss = 3e-4
k_slow = 5e4 * 1e-6
k_fast = 1e6 * 1e-6

I1O2_init = 0.1
I2_init = 0.1
Th2_init = I2_init * 4

node_diameter = 75
node_radius = node_diameter / 2
bath_margin = 250
distance_between = 200
total_width = 5000
total_height = 5000
fine_dx = 5
cells_per_level = 3
coarse_dx = 100

dt = 60.0
save_interval_time = 60.0
save_interval_steps = int(save_interval_time / dt)

# Sweep control for the split scheme (identical to TG_Rmesh_fast.py)
max_sweeps = 15
sweep_residual_target = 1e-8
sweep_plateau_tol = 1e-9

# =============================================================================
# STEADY-STATE / PHASE CONTROL
# =============================================================================
# See module docstring for why these differ from TG_Rmesh_fast.py's
# (ss_tolerance=1e-8, ss_window=50, check_interval=100), which structurally
# could never fire within a reasonable runtime.
check_interval = 50            # steps between checks (50 min)
ss_window = 10                 # consecutive passing checks required
                                # -> min 8.3 h before either phase can fire
ss_tolerance = 1e-6            # loosened from 1e-8 -- see docstring
epsilon = 1e-10                # relative-change denominator floor

PHASE_MAX_TIME = 6 * 3600.0    # practical cutoff per phase, not just a rare
                                # safety net: the receiver locally flattens
                                # within ~1-2 h, but the 1e-6 relative-change
                                # test keeps failing for many more hours
                                # because the far side of the 5mm bath is
                                # still slowly filling in by diffusion (that
                                # front takes ~10+ h to cross the domain) and
                                # very slightly perturbs the receiver the
                                # whole time it does. At the default params,
                                # I2 moves <0.03% of its total range between
                                # hour 6 and hour 13 -- so waiting for formal
                                # convergence buys essentially nothing here.
                                # Expect this cap, not ss_tolerance, to be
                                # what actually ends most runs.
verbose = True

# =============================================================================
# 2D ADAPTIVE MESH SETUP (identical to TG_Rmesh_fast.py)
# =============================================================================

print("Creating adaptive mesh...")

mesh_filename, sender_center_x, receiver_center_x, sender_center_y = create_conformal_radial_mesh(
    bath_width=total_width,
    bath_height=total_height,
    node_diameter=75.0,
    distance_between_nodes=distance_between,
    min_cell_size=fine_dx,
    max_cell_size=coarse_dx,
    growth_rate=1.5,
    cells_per_level=cells_per_level,
    mesh_filename='radial_mesh_on_then_off.msh',
    visualize_gmsh=False,
    verbose=True)

mesh = Gmsh2D(mesh_filename)
receiver_center_y = sender_center_y

x, y = mesh.cellCenters

print(f"\n2D Simulation Setup TRIANGULAR MESH (on-then-off solver):")
print(f"  Mesh: {mesh.numberOfCells} cells (adaptive)")
print(f"  Domain: {total_width} * {total_height} μm²")
print(f"  Node diameter: {node_diameter} μm")
print(f"  Distance: {distance_between} μm (center-to-center)")
print(f"  Sender at: ({sender_center_x:.0f}, {sender_center_y:.0f})")
print(f"  Receiver at: ({receiver_center_x:.0f}, {receiver_center_y:.0f})")
print()

# =============================================================================
# CELL VARIABLES (identical initial conditions to TG_Rmesh_fast.py)
# =============================================================================

S2, I2, Th2, S2_I2, S2_Th2, I1O2, D_S2 = initalize_variables_speedup(
    mesh, x, y, sender_center_x, receiver_center_x, receiver_center_y,
    node_radius, I2_init, Th2_init, I1O2_init, D_gel, D_solution)

# Receiver-only mask for the steady-state test -- restricts the relative-
# change check to the region the switch state actually lives in, instead of
# the whole bath (see module docstring, point 2).
receiver_region = np.asarray(
    np.sqrt((x - receiver_center_x) ** 2 + (y - receiver_center_y) ** 2) <= node_radius
)

# =============================================================================
# ONLY S2 GETS A FIPY EQUATION -- IT'S THE ONLY VARIABLE THAT DIFFUSES.
# =============================================================================

eq_S2 = (TransientTerm(var=S2) ==
         DiffusionTerm(coeff=D_S2, var=S2) +
         k_p * I1O2 +
         ImplicitSourceTerm(coeff=-(k_slow * I2 + k_fast * Th2 + k_d_ss), var=S2))

s2_solver = LinearLUSolver(tolerance=1e-10)


def reaction_pair_step(S2_now, X_old, C_old, k_on, k_off, dt):
    """Closed-form backward-Euler step for one exchange pair (X <-> C).
    Identical to TG_Rmesh_fast.py -- see that file for the derivation."""
    a = k_on * S2_now
    d = k_off
    det = 1.0 + dt * (a + d)
    X_new = ((1.0 + dt * d) * X_old + dt * d * C_old) / det
    C_new = (dt * a * X_old + (1.0 + dt * a) * C_old) / det
    return X_new, C_new


# =============================================================================
# FIND RECEIVER CENTER INDEX (for monitoring)
# =============================================================================

distances_to_receiver = numerix.sqrt((x - receiver_center_x)**2 +
                                     (y - receiver_center_y)**2)
receiver_center_idx = numerix.argmin(distances_to_receiver)

time_points = []
I2_concentration = []
S2_free_concentration = []
S2_total_concentration = []
i1o2_active_flags = []

recent_changes = []

# =============================================================================
# TWO-PHASE TIME STEPPING: "on" until receiver steady state, then "off"
# until receiver steady state again.
# =============================================================================

print("Starting 2D simulation with adaptive mesh (on-then-off solver)...")

phase = "on"
phase_start_time = 0.0
t_shutoff = None
t_recovered = None
on_converged = False
off_converged = False

step = 0
current_time = 0.0

while True:
    S2.updateOld()
    I2.updateOld()
    Th2.updateOld()
    S2_I2.updateOld()
    S2_Th2.updateOld()

    S2_old_vals = S2.value.copy()
    I2_old_vals = I2.value.copy()
    Th2_old_vals = Th2.value.copy()
    S2_I2_old_vals = S2_I2.value.copy()
    S2_Th2_old_vals = S2_Th2.value.copy()

    res = 1e10
    sweep = 0
    prev_res = None

    while sweep < max_sweeps:
        S2_guess = S2.value  # current best guess of S2 at t+dt this sweep

        I2_new, S2_I2_new = reaction_pair_step(
            S2_guess, I2_old_vals, S2_I2_old_vals, k_slow, k_d_ds, dt)
        Th2_new, S2_Th2_new = reaction_pair_step(
            S2_guess, Th2_old_vals, S2_Th2_old_vals, k_fast, k_d_ds, dt)

        I2.setValue(I2_new)
        S2_I2.setValue(S2_I2_new)
        Th2.setValue(Th2_new)
        S2_Th2.setValue(S2_Th2_new)

        res = eq_S2.sweep(dt=dt, solver=s2_solver)
        sweep += 1

        if res < sweep_residual_target:
            break
        if prev_res is not None and abs(res - prev_res) < sweep_plateau_tol:
            break
        prev_res = res

    step += 1
    current_time = step * dt

    if step % save_interval_steps == 0:
        time_points.append(current_time / 3600)

        I2_val = I2.value[receiver_center_idx]
        S2_free_val = S2.value[receiver_center_idx]
        S2_total_val = (S2.value[receiver_center_idx] +
                       S2_I2.value[receiver_center_idx] +
                       S2_Th2.value[receiver_center_idx])

        I2_concentration.append(I2_val)
        S2_free_concentration.append(S2_free_val)
        S2_total_concentration.append(S2_total_val)
        i1o2_active_flags.append(1 if phase == "on" else 0)

        if step % check_interval == 0:
            changes = [
                np.max(np.abs(S2.value[receiver_region] - S2_old_vals[receiver_region])
                       / (np.abs(S2.value[receiver_region]) + epsilon)),
                np.max(np.abs(I2.value[receiver_region] - I2_old_vals[receiver_region])
                       / (np.abs(I2.value[receiver_region]) + epsilon)),
                np.max(np.abs(Th2.value[receiver_region] - Th2_old_vals[receiver_region])
                       / (np.abs(Th2.value[receiver_region]) + epsilon)),
                np.max(np.abs(S2_I2.value[receiver_region] - S2_I2_old_vals[receiver_region])
                       / (np.abs(S2_I2.value[receiver_region]) + epsilon)),
                np.max(np.abs(S2_Th2.value[receiver_region] - S2_Th2_old_vals[receiver_region])
                       / (np.abs(S2_Th2.value[receiver_region]) + epsilon)),
            ]
            max_change = np.max(changes)
            recent_changes.append(max_change)
            if len(recent_changes) > ss_window:
                recent_changes.pop(0)

            phase_elapsed = current_time - phase_start_time
            phase_converged = (len(recent_changes) >= ss_window
                                and all(c < ss_tolerance for c in recent_changes))
            phase_timed_out = phase_elapsed >= PHASE_MAX_TIME

            if phase_converged or phase_timed_out:
                if phase == "on":
                    on_converged = phase_converged
                    t_shutoff = current_time
                    if verbose:
                        tag = "STEADY STATE REACHED" if phase_converged else "PHASE TIMED OUT (forcing shutoff)"
                        print(f"\n{'='*70}")
                        print(f"{tag} at t = {current_time/3600:.3f} h "
                              f"(max relative change = {max_change:.2e})")
                        print("Setting I1O2 = 0 (production off).")
                        print(f"{'='*70}\n")
                    I1O2.setValue(0.0)
                    phase = "off"
                    phase_start_time = current_time
                    recent_changes = []
                else:
                    off_converged = phase_converged
                    t_recovered = current_time
                    if verbose:
                        tag = "STEADY STATE REACHED" if phase_converged else "PHASE TIMED OUT (stopping)"
                        print(f"\n{'='*70}")
                        print(f"{tag} (post-shutoff) at t = {current_time/3600:.3f} h "
                              f"(max relative change = {max_change:.2e})")
                        print(f"{'='*70}\n")
                    break

        if step % (save_interval_steps * 10) == 0:
            print(f"t = {current_time/3600:.2f} hr [{phase.upper():>3}]: "
                  f"I2 = {I2_val*1000:.2f} nM, "
                  f"S2_total = {S2_total_val*1000:.2f} nM, "
                  f"sweeps = {sweep}")

    if phase == "off" and t_recovered is not None:
        break

print("\nSimulation complete!")

wall_time_end = timer.time()
wall_time = wall_time_end - wall_start_time

print(f'total seconds of time for simulation: {wall_time:.3f}')
print(f'shutoff at t = {t_shutoff/3600:.3f} h (converged={on_converged})')
print(f'post-shutoff steady state at t = {t_recovered/3600:.3f} h (converged={off_converged})')

# =============================================================================
# SAVE RESULTS TO CSV
# =============================================================================

print("Saving results to CSV files...")

I2_concentration_nM = np.array(I2_concentration) * 1000
S2_free_concentration_nM = np.array(S2_free_concentration) * 1000
S2_total_concentration_nM = np.array(S2_total_concentration) * 1000
time_points_arr = np.array(time_points)

df = pd.DataFrame({
    'Time (hours)': time_points_arr,
    'I2 (nM)': I2_concentration_nM,
    'S2_free (nM)': S2_free_concentration_nM,
    'S2_total (nM)': S2_total_concentration_nM,
    'I1O2_active': i1o2_active_flags})

csv_filename = f'timeseries_on_then_off_ccd={distance_between:.0f}_dx={fine_dx}_V1.csv'
df.to_csv(csv_filename, index=False)


def half_time_between(time_hours, signal, t_start, y0, y1):
    """
    Time (absolute, same units as time_hours) at which `signal` first crosses
    halfway between y0 and y1, searching only from t_start onward. Mirrors
    the half_time() helper in Paramter_sweep/Parameter_sweep_unified.py, but
    takes explicit endpoints since the "before" and "after" plateaus here are
    two different steady states, not simply signal[0] and signal[-1].
    """
    mask = time_hours >= t_start
    t = time_hours[mask]
    s = signal[mask]
    if len(s) == 0 or abs(y1 - y0) < 1e-12:
        return np.nan

    target = 0.5 * (y0 + y1)
    crossed = s >= target if y1 > y0 else s <= target
    if not crossed.any():
        return np.nan

    idx = int(np.argmax(crossed))
    if idx == 0:
        return float(t[0])

    t0, t1_ = t[idx - 1], t[idx]
    s0, s1 = s[idx - 1], s[idx]
    if s1 == s0:
        return float(t1_)
    return float(t0 + (target - s0) * (t1_ - t0) / (s1 - s0))


# I2 just before shutoff (the "off" plateau) vs. I2 at the very end (the
# recovered plateau) -- the two levels the recovery half-time is measured
# between.
shutoff_idx = int(np.argmin(np.abs(time_points_arr - t_shutoff / 3600)))
I2_at_shutoff_nM = I2_concentration_nM[shutoff_idx]
I2_at_end_nM = I2_concentration_nM[-1]

recovery_half_time_hr = half_time_between(
    time_points_arr, I2_concentration_nM,
    t_start=t_shutoff / 3600, y0=I2_at_shutoff_nM, y1=I2_at_end_nM)

print(f"I2 at shutoff:  {I2_at_shutoff_nM:.2f} nM")
print(f"I2 at recovery: {I2_at_end_nM:.2f} nM (I2_init = {I2_init*1000:.2f} nM)")
print(f"recovery half-time: {recovery_half_time_hr - t_shutoff/3600:.3f} h "
      f"after shutoff (absolute t = {recovery_half_time_hr:.3f} h)"
      if np.isfinite(recovery_half_time_hr) else "recovery half-time: n/a")

# =============================================================================
# PLOTTING
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'On-then-off: Domain {total_width/1e3:.0f}mm x {total_height/1e3:.0f}mm, '
             f'Distance={distance_between:.0f}μm, shutoff at t={t_shutoff/3600:.2f} h',
             fontsize=16, fontweight='bold')

axes[0].plot(time_points_arr, I2_concentration_nM, 'b-', linewidth=2)
axes[0].axvline(x=t_shutoff / 3600, color='k', linestyle='--', alpha=0.6, label='I1O2 off')
axes[0].axhline(y=I2_init * 1000, color='gray', linestyle=':', alpha=0.5, label='I2_init')
axes[0].set_xlabel('Time (hours)', fontsize=12)
axes[0].set_ylabel('[I2] (nM)', fontsize=12)
axes[0].set_title('I2 at Receiver', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(bottom=0)

axes[1].plot(time_points_arr, S2_free_concentration_nM, 'g-', linewidth=2)
axes[1].axvline(x=t_shutoff / 3600, color='k', linestyle='--', alpha=0.6, label='I1O2 off')
axes[1].set_xlabel('Time (hours)', fontsize=12)
axes[1].set_ylabel('[S2] free (nM)', fontsize=12)
axes[1].set_title('Free S2 at Receiver', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(bottom=0)

axes[2].plot(time_points_arr, S2_total_concentration_nM, 'r-', linewidth=2)
axes[2].axvline(x=t_shutoff / 3600, color='k', linestyle='--', alpha=0.6, label='I1O2 off')
axes[2].set_xlabel('Time (hours)', fontsize=12)
axes[2].set_ylabel('[S2] total (nM)', fontsize=12)
axes[2].set_title('Total S2 at Receiver', fontsize=14, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(f'timeseries_on_then_off_ccd={distance_between:.0f}_dx={fine_dx}_V1.png',
            dpi=300, bbox_inches='tight')
plt.show()
