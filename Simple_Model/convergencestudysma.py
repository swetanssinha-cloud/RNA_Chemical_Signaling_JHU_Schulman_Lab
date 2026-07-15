# WHAT I WAS GOING TO WRITE MYSELF

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sma_changeKp import run_simulation

# params = {
#     'k_slow': 1e5,      # M^-1 s^-1
#     'k_fast': 1e6,      # M^-1 s^-1
#     'k_d_ds': 3e-4,     # s^-1
#     'k_d_ss': 3e-4,     # s^-1
#     'I2_0': 100e-9,     # 100 nM in M
#     'Th2_0': 5000e-9,   # 5000 nM in M      
#     'n_steps': 480,     # 8 hours
#     'max_sweeps': 20,
#     'tol': 1e-10
# }

# dt_values = np.array([2,1,0.5,0.25]) * 60 #saying 2 minutes, 1 minute, 30 seconds, 15 seconds as time step 

# for dt in dt_values:
#     print(f"\n{'='*50}")
#     print(f"Running with dx = {dt}")
#     print(f"{'='*50}")

#     total_time = 8 * 3600

#WHAT CHAT WROTE FOR ME INSTEAD

import numpy as np
import matplotlib.pyplot as plt
from fipy import Grid1D, CellVariable, TransientTerm, ImplicitSourceTerm
import time

# ============================================================================
# PARAMETERS
# ============================================================================
k_slow = 1e5   # M^-1 s^-1
k_fast = 1e6   # M^-1 s^-1
k_d_ds = 3e-4   # s^-1
k_d_ss = 3e-4   # s^-1

I2_0 = 100e-9   # M
Th2_0 = 5e-6    # M
S2_0 = 0.0
S2_I2_0 = 0.0
S2_Th2_0 = 0.0

Phi_in = 0.1e-9  # M/s

t_end = 8 * 3600  # 8 hours
nonlinear_tolerance = 1e-9
max_sweeps_per_step = 20

# Choose the dt values to test
dt_values = [60.0, 30.0, 15.0, 10.0, 5.0]   # seconds

results = {}

print("=" * 70)
print("TIME STEP CONVERGENCE STUDY")
print("=" * 70)
print(f"Total simulation time: {t_end/3600:.2f} hours")
print(f"Nonlinear tolerance: {nonlinear_tolerance}")
print(f"Max sweeps per step: {max_sweeps_per_step}")
print()

# ============================================================================
# HELPER FUNCTION TO RUN ONE SIMULATION
# ============================================================================
def run_simulation_for_dt(dt):
    """
    Run the FiPy model for a single time step size dt.
    Returns final concentrations and diagnostics.
    """
    n_steps = int(np.ceil(t_end / dt))
    actual_t_end = n_steps * dt

    # One-cell mesh
    mesh = Grid1D(nx=1, dx=1.0)

    # Variables
    S2 = CellVariable(name="S2", mesh=mesh, hasOld=True, value=S2_0)
    I2 = CellVariable(name="I2", mesh=mesh, hasOld=True, value=I2_0)
    S2_I2 = CellVariable(name="S2_I2", mesh=mesh, hasOld=True, value=S2_I2_0)
    Th2 = CellVariable(name="Th2", mesh=mesh, hasOld=True, value=Th2_0)
    S2_Th2 = CellVariable(name="S2_Th2", mesh=mesh, hasOld=True, value=S2_Th2_0)

    # Equations
    eq_S2 = (TransientTerm(var=S2) ==
             Phi_in +
             ImplicitSourceTerm(coeff=-k_slow * I2, var=S2) +
             ImplicitSourceTerm(coeff=-k_fast * Th2, var=S2) +
             ImplicitSourceTerm(coeff=-k_d_ss, var=S2))

    eq_I2 = (TransientTerm(var=I2) ==
             k_d_ds * S2_I2 +
             ImplicitSourceTerm(coeff=-k_slow * S2, var=I2))

    eq_Th2 = (TransientTerm(var=Th2) ==
              k_d_ds * S2_Th2 +
              ImplicitSourceTerm(coeff=-k_fast * S2, var=Th2))

    eq_S2_I2 = (TransientTerm(var=S2_I2) ==
                k_slow * I2 * S2 +
                ImplicitSourceTerm(coeff=-k_d_ds, var=S2_I2))

    eq_S2_Th2 = (TransientTerm(var=S2_Th2) ==
                 k_fast * Th2 * S2 +
                 ImplicitSourceTerm(coeff=-k_d_ds, var=S2_Th2))

    def clip_nonnegative():
        S2.setValue(np.maximum(np.asarray(S2.value), 0.0))
        I2.setValue(np.maximum(np.asarray(I2.value), 0.0))
        S2_I2.setValue(np.maximum(np.asarray(S2_I2.value), 0.0))
        Th2.setValue(np.maximum(np.asarray(Th2.value), 0.0))
        S2_Th2.setValue(np.maximum(np.asarray(S2_Th2.value), 0.0))

    total_sweeps = 0
    start_time = time.time()

    for step in range(1, n_steps + 1):
        # Update old values
        S2.updateOld()
        I2.updateOld()
        S2_I2.updateOld()
        Th2.updateOld()
        S2_Th2.updateOld()

        residual = np.inf
        sweep_count = 0

        while residual > nonlinear_tolerance and sweep_count < max_sweeps_per_step:
            residual = 0.0
            residual = max(residual, eq_S2.sweep(var=S2, dt=dt))
            residual = max(residual, eq_I2.sweep(var=I2, dt=dt))
            residual = max(residual, eq_S2_I2.sweep(var=S2_I2, dt=dt))
            residual = max(residual, eq_Th2.sweep(var=Th2, dt=dt))
            residual = max(residual, eq_S2_Th2.sweep(var=S2_Th2, dt=dt))
            clip_nonnegative()
            sweep_count += 1
            total_sweeps += 1

    elapsed = time.time() - start_time

    return {
        "dt": dt,
        "n_steps": n_steps,
        "actual_t_end": actual_t_end,
        "runtime": elapsed,
        "total_sweeps": total_sweeps,
        "avg_sweeps_per_step": total_sweeps / n_steps,
        "S2_final": float(S2.value[0]),
        "I2_final": float(I2.value[0]),
        "S2_I2_final": float(S2_I2.value[0]),
        "Th2_final": float(Th2.value[0]),
        "S2_Th2_final": float(S2_Th2.value[0]),
    }

# ============================================================================
# RUN CONVERGENCE STUDY
# ============================================================================
for dt in dt_values:
    print(f"\n{'='*60}")
    print(f"Running with dt = {dt} s")
    print(f"{'='*60}")

    res = run_simulation_for_dt(dt)
    results[dt] = res

    print(f"  n_steps = {res['n_steps']}")
    print(f"  actual_t_end = {res['actual_t_end']/3600:.4f} h")
    print(f"  runtime = {res['runtime']:.2f} s")
    print(f"  avg sweeps/step = {res['avg_sweeps_per_step']:.2f}")
    print(f"  Final S2 = {res['S2_final']*1e9:.4f} nM")
    print(f"  Final I2 = {res['I2_final']*1e9:.4f} nM")
    print(f"  Final S2:I2 = {res['S2_I2_final']*1e9:.4f} nM")
    print(f"  Final Th2 = {res['Th2_final']*1e9:.4f} nM")
    print(f"  Final S2:Th2 = {res['S2_Th2_final']*1e9:.4f} nM")

# ============================================================================
# PRINT SUMMARY TABLE
# ============================================================================
print("\n" + "="*90)
print("TIME STEP CONVERGENCE RESULTS")
print("="*90)
print(f"{'dt (s)':<10} {'n_steps':<10} {'S2 (nM)':<12} {'I2 (nM)':<12} {'S2:I2 (nM)':<14} "
      f"{'Th2 (nM)':<12} {'S2:Th2 (nM)':<14} {'runtime (s)':<12}")
print("-" * 90)

for dt in dt_values:
    r = results[dt]
    print(f"{dt:<10.2f} {r['n_steps']:<10d} "
          f"{r['S2_final']*1e9:<12.4f} {r['I2_final']*1e9:<12.4f} "
          f"{r['S2_I2_final']*1e9:<14.4f} {r['Th2_final']*1e9:<12.4f} "
          f"{r['S2_Th2_final']*1e9:<14.4f} {r['runtime']:<12.2f}")

# ============================================================================
# CONVERGENCE METRICS
# ============================================================================
print("\n" + "="*90)
print("CONVERGENCE METRICS (% change between successive refinements)")
print("="*90)

for i in range(len(dt_values) - 1):
    dt_coarse = dt_values[i]
    dt_fine = dt_values[i + 1]

    coarse = results[dt_coarse]
    fine = results[dt_fine]

    def pct_change(a, b):
        if a == 0:
            return np.nan
        return abs(b - a) / abs(a) * 100

    S2_change = pct_change(coarse["S2_final"], fine["S2_final"])
    I2_change = pct_change(coarse["I2_final"], fine["I2_final"])
    S2I2_change = pct_change(coarse["S2_I2_final"], fine["S2_I2_final"])
    Th2_change = pct_change(coarse["Th2_final"], fine["Th2_final"])
    S2Th2_change = pct_change(coarse["S2_Th2_final"], fine["S2_Th2_final"])

    print(f"\nRefining from dt = {dt_coarse} s to dt = {dt_fine} s:")
    print(f"  S2 change:      {S2_change:.3f}%")
    print(f"  I2 change:      {I2_change:.3f}%")
    print(f"  S2:I2 change:   {S2I2_change:.3f}%")
    print(f"  Th2 change:     {Th2_change:.3f}%")
    print(f"  S2:Th2 change:  {S2Th2_change:.3f}%")

# Check convergence using the finest two dt values on S2 total RNA
finest_dt = dt_values[-1]
prev_dt = dt_values[-2]
finest_change = abs(results[finest_dt]["S2_final"] - results[prev_dt]["S2_final"]) / abs(results[prev_dt]["S2_final"]) * 100 if results[prev_dt]["S2_final"] != 0 else np.nan

print("\n" + "="*90)
if finest_change < 1.0:
    print(f"✓ CONVERGED: Change in S2 between dt={prev_dt} and dt={finest_dt} is {finest_change:.3f}% < 1%")
else:
    print(f"✗ NOT CONVERGED: Change in S2 between dt={prev_dt} and dt={finest_dt} is {finest_change:.3f}% > 1%")
    print("  (You may still consider this acceptable depending on your accuracy needs.)")
print("="*90)

# ============================================================================
# PLOTTING
# ============================================================================
dt_plot = np.array(dt_values)

S2_vals = np.array([results[dt]["S2_final"] for dt in dt_values]) * 1e9
I2_vals = np.array([results[dt]["I2_final"] for dt in dt_values]) * 1e9
S2I2_vals = np.array([results[dt]["S2_I2_final"] for dt in dt_values]) * 1e9
Th2_vals = np.array([results[dt]["Th2_final"] for dt in dt_values]) * 1e9
S2Th2_vals = np.array([results[dt]["S2_Th2_final"] for dt in dt_values]) * 1e9

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Total free S2
axes[0].plot(dt_plot, S2_vals, 'o-', linewidth=2, markersize=8)
axes[0].set_xscale('log')
axes[0].invert_xaxis()
axes[0].set_xlabel('Time step dt (s)')
axes[0].set_ylabel('Final S2 (nM)')
axes[0].set_title('S2 Convergence')
axes[0].grid(True)

# Complexes
axes[1].plot(dt_plot, S2I2_vals, 'o-', linewidth=2, markersize=8, label='S2:I2')
axes[1].plot(dt_plot, S2Th2_vals, 's--', linewidth=2, markersize=8, label='S2:Th2')
axes[1].set_xscale('log')
axes[1].invert_xaxis()
axes[1].set_xlabel('Time step dt (s)')
axes[1].set_ylabel('Final complex concentration (nM)')
axes[1].set_title('Complex Convergence')
axes[1].grid(True)
axes[1].legend()

# Free I2 and Th2
axes[2].plot(dt_plot, I2_vals, 'o-', linewidth=2, markersize=8, label='I2')
axes[2].plot(dt_plot, Th2_vals, 's--', linewidth=2, markersize=8, label='Th2')
axes[2].set_xscale('log')
axes[2].invert_xaxis()
axes[2].set_xlabel('Time step dt (s)')
axes[2].set_ylabel('Final free concentration (nM)')
axes[2].set_title('Free Species Convergence')
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.savefig('time_step_convergence_study.png', dpi=300, bbox_inches='tight')
print("\nConvergence plot saved as 'time_step_convergence_study.png'")
plt.show()



