import numpy as np
import matplotlib.pyplot as plt
from fipy import Grid1D, CellVariable, TransientTerm, ImplicitSourceTerm
import time

'''

'''

# ============================================================================
# PARAMETERS (matching original defaults)
# ============================================================================
k_slow = 1e5  # M^-1 s^-1
k_fast = 1e6  # M^-1 s^-1
k_d_ds = 3e-4  # s^-1
k_d_ss = 3e-4  # s^-1

I2_0 = 100e-9   # M
Th2_0 = 5e-6    # M
S2_0 = 0.0
S2_I2_0 = 0.0
S2_Th2_0 = 0.0

#===============================

Phi_in = 0.1e-9  # M/s

#==============================

# TIME PARAMETERS - MATCHING ORIGINAL
dt = 10 #60.0  
t_end = 8 * 3600
n_steps = int(np.ceil(t_end / dt))  # 480 steps

nonlinear_tolerance = 1e-9
max_sweeps_per_step = 20

print("="*70)
print("OPTION A - FiPy Implementation (Matching Original Style)")
#print("TESTING TESTING=================================="")
print("="*70)
print(f"Timestep: dt = {dt} s ")
print(f"Duration: {t_end/3600} hours")
print(f"Total steps: {n_steps}")
print(f"Max sweeps per step: {max_sweeps_per_step}")
print(f"Tolerance: {nonlinear_tolerance}")
print()

# ============================================================================
# MESH AND VARIABLES
# ============================================================================
mesh = Grid1D(nx=1, dx=1.0)

S2 = CellVariable(name="S2", mesh=mesh, hasOld=True, value=S2_0)
I2 = CellVariable(name="I2", mesh=mesh, hasOld=True, value=I2_0)
S2_I2 = CellVariable(name="S2_I2", mesh=mesh, hasOld=True, value=S2_I2_0)
Th2 = CellVariable(name="Th2", mesh=mesh, hasOld=True, value=Th2_0)
S2_Th2 = CellVariable(name="S2_Th2", mesh=mesh, hasOld=True, value=S2_Th2_0)

# ============================================================================
# EQUATIONS (matching original structure)
# ============================================================================

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

# ============================================================================
# STORAGE
# ============================================================================


times = np.zeros(n_steps + 1)
S2_history = np.zeros(n_steps + 1)
I2_history = np.zeros(n_steps + 1)
S2_I2_history = np.zeros(n_steps + 1)
Th2_history = np.zeros(n_steps + 1)
S2_Th2_history = np.zeros(n_steps + 1)

times[0] = 0.0
S2_history[0] = S2.value[0]
I2_history[0] = I2.value[0]
S2_I2_history[0] = S2_I2.value[0]
Th2_history[0] = Th2.value[0]
S2_Th2_history[0] = S2_Th2.value[0]

# ============================================================================
# TIME STEPPING (matching original loop structure)
# ============================================================================
def clip_nonnegative():
    """Enforce non-negativity like original code."""
    S2.setValue(np.maximum(np.asarray(S2.value), 0.0))
    I2.setValue(np.maximum(np.asarray(I2.value), 0.0))
    S2_I2.setValue(np.maximum(np.asarray(S2_I2.value), 0.0))
    Th2.setValue(np.maximum(np.asarray(Th2.value), 0.0))
    S2_Th2.setValue(np.maximum(np.asarray(S2_Th2.value), 0.0))


start_time = time.time()
total_sweeps = 0

print("Starting time integration...")
print()

for step in range(1, n_steps + 1):
    # Update old values (like original)
    S2.updateOld()
    I2.updateOld()
    S2_I2.updateOld()
    Th2.updateOld()
    S2_Th2.updateOld()
    
    # Sweep until convergence (like original)
    residual = np.inf
    sweep_count = 0
    
    while residual > nonlinear_tolerance and sweep_count < max_sweeps_per_step:
        residual = 0.0
        # Sweep each equation separately (like original)
        residual = max(residual, eq_S2.sweep(var=S2, dt=dt))
        residual = max(residual, eq_I2.sweep(var=I2, dt=dt))
        residual = max(residual, eq_S2_I2.sweep(var=S2_I2, dt=dt))
        residual = max(residual, eq_Th2.sweep(var=Th2, dt=dt))
        residual = max(residual, eq_S2_Th2.sweep(var=S2_Th2, dt=dt))
        clip_nonnegative()
        sweep_count += 1
        total_sweeps += 1
    
    # Store results
    times[step] = step * dt
    S2_history[step] = S2.value[0]
    I2_history[step] = I2.value[0]
    S2_I2_history[step] = S2_I2.value[0]
    Th2_history[step] = Th2.value[0]
    S2_Th2_history[step] = S2_Th2.value[0]
    
    # Print progress (like original: every 10% plus first/last)
    if step == 1 or step % max(1, n_steps // 10) == 0 or step == n_steps:
        hour = step * dt / 3600
        print(f"step {step:4d}/{n_steps} | t = {hour:5.2f} h | "
              f"S2 = {S2.value[0]*1e9:8.3f} nM | "
              f"I2 = {I2.value[0]*1e9:8.3f} nM | "
              f"sweeps = {sweep_count:2d} | residual = {residual:.3e}")

elapsed = time.time() - start_time

print()
print("="*70)
print("SIMULATION COMPLETE")
print("="*70)
print(f"Total runtime: {elapsed:.2f} seconds")
print(f"Total sweeps: {total_sweeps}")
print(f"Average sweeps per step: {total_sweeps/n_steps:.1f}")
print(f"Steps per second: {n_steps/elapsed:.1f}")
print()

# ============================================================================
# FINAL DIAGNOSTICS
# ============================================================================

print("Final concentrations (t = 8 hours):")
print(f"  S2     = {S2.value[0]*1e9:.3f} nM")
print(f"  I2     = {I2.value[0]*1e9:.3f} nM")
print(f"  S2:I2  = {S2_I2.value[0]*1e9:.3f} nM")
print(f"  Th2    = {Th2.value[0]*1e9:.1f} nM")
print(f"  S2:Th2 = {S2_Th2.value[0]*1e9:.1f} nM")
print()

total_rna = (S2.value[0] + S2_I2.value[0] + S2_Th2.value[0]) * 1e9
print(f"Receiver total RNA: {total_rna:.3f} nM")
print()

# ============================================================================
# PLOTTING (matching original style)
# ============================================================================
t_hours = times / 3600
S2_nM = S2_history * 1e9
I2_nM = I2_history * 1e9
S2_I2_nM = S2_I2_history * 1e9
Th2_nM = Th2_history * 1e9
S2_Th2_nM = S2_Th2_history * 1e9
total_rna_nM = S2_nM + S2_I2_nM + S2_Th2_nM


plt.figure()
plt.plot(times, S2_nM, label='S2 (free)', color='blue')
plt.plot(times, S2_I2_nM, label='S2:I2 (slow)', color='orange')
plt.plot(times, S2_Th2_nM, label='S2:Th2 (fast)', color='green')
plt.show