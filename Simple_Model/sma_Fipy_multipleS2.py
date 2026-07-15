import numpy as np
from fipy import CellVariable, TransientTerm, ImplicitSourceTerm, Grid1D
import matplotlib.pyplot as plt

def run_simulation(Phi_in_value, params):
    """
    Run well-mixed FiPy simulation for given Phi_in value.
    Returns time and S2 arrays.
    """
    # Unpack parameters
    k_slow = params['k_slow']
    k_fast = params['k_fast']
    k_d_ds = params['k_d_ds']
    k_d_ss = params['k_d_ss']
    I2_0 = params['I2_0']
    Th2_0 = params['Th2_0']
    dt_s = params['dt_s']
    n_steps = params['n_steps']
    max_sweeps = params['max_sweeps']
    tol = params['tol']
    
    # Create 1-cell mesh
    mesh = Grid1D(nx=1, dx=1.0)
    
    # Initialize CellVariables (fresh for each simulation)
    S2 = CellVariable(name="S2", mesh=mesh, value=0.0, hasOld=True)
    I2 = CellVariable(name="I2", mesh=mesh, value=I2_0, hasOld=True)
    Th2 = CellVariable(name="Th2", mesh=mesh, value=Th2_0, hasOld=True)
    C_I2 = CellVariable(name="C_I2", mesh=mesh, value=0.0, hasOld=True)
    C_Th2 = CellVariable(name="C_Th2", mesh=mesh, value=0.0, hasOld=True)
    
    # Define equations (well-mixed: no DiffusionTerm)
    eq_S2 = (TransientTerm(var=S2) == 
             ImplicitSourceTerm(coeff=-k_slow * I2, var=S2) +
             ImplicitSourceTerm(coeff=-k_fast * Th2, var=S2) +
             ImplicitSourceTerm(coeff=-k_d_ss, var=S2) +
             Phi_in_value)
    
    eq_I2 = (TransientTerm(var=I2) == 
             ImplicitSourceTerm(coeff=-k_slow * S2, var=I2))
    
    eq_Th2 = (TransientTerm(var=Th2) == 
              ImplicitSourceTerm(coeff=-k_fast * S2, var=Th2))
    
    eq_C_I2 = (TransientTerm(var=C_I2) == 
               k_slow * S2 * I2 +
               ImplicitSourceTerm(coeff=-k_d_ds, var=C_I2))
    
    eq_C_Th2 = (TransientTerm(var=C_Th2) == 
                k_fast * S2 * Th2 +
                ImplicitSourceTerm(coeff=-k_d_ds, var=C_Th2))
    
    # Storage
    time_history = []
    S2_history = []
    
    # Time stepping
    for step in range(n_steps):
        # Update oldValue for all variables
        S2.updateOld()
        I2.updateOld()
        Th2.updateOld()
        C_I2.updateOld()
        C_Th2.updateOld()
        
        # Sweep equations until convergence
        for sweep in range(max_sweeps):
            res_S2 = eq_S2.sweep(dt=dt_s)
            res_I2 = eq_I2.sweep(dt=dt_s)
            res_Th2 = eq_Th2.sweep(dt=dt_s)
            res_C_I2 = eq_C_I2.sweep(dt=dt_s)
            res_C_Th2 = eq_C_Th2.sweep(dt=dt_s)
            
            residual = max(res_S2, res_I2, res_Th2, res_C_I2, res_C_Th2)
            
            if residual < tol:
                break
        
        # Store current state
        current_time = (step + 1) * dt_s
        time_history.append(current_time)
        S2_history.append(S2.value[0])
    
    return np.array(time_history), np.array(S2_history)


# Parameters (matching original paper)
params = {
    'k_slow': 1e5,      # M^-1 s^-1
    'k_fast': 1e6,      # M^-1 s^-1
    'k_d_ds': 3e-4,     # s^-1
    'k_d_ss': 3e-4,     # s^-1
    'I2_0': 100e-9,     # 100 nM in M
    'Th2_0': 5000e-9,   # 5000 nM in M
    'dt_s': 60.0,       # 60 second timestep
    'n_steps': 480,     # 8 hours
    'max_sweeps': 20,
    'tol': 1e-10
}

# Array of Phi_in values to sweep (in M/s)
Phi_in_array = np.array([0.02, 0.05, 0.1, 0.2, 0.5]) * 1e-9  # nM/s → M/s

Phi_in_array = np.array([1, 2, 3, 4, 5]) * 1e-9 

# Run simulations and collect results
results = {}
for Phi_in_val in Phi_in_array:
    print(f"Running simulation for Phi_in = {Phi_in_val*1e9:.2f} nM/s...")
    time, S2 = run_simulation(Phi_in_val, params)
    results[Phi_in_val] = (time, S2)

# Plot all curves on single figure
plt.figure(figsize=(10, 6))
for Phi_in_val in Phi_in_array:
    time, S2 = results[Phi_in_val]
    plt.plot(time / 3600, S2 * 1e9, 
             label=f'Φ_in = {Phi_in_val*1e9:.2f} nM/s', 
             linewidth=2)

plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('[S2] (nM)', fontsize=12)
plt.title('Signal S2 vs Time for Different Input Fluxes', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print final steady-state values
print("\nFinal [S2] values at t=8h:")
for Phi_in_val in Phi_in_array:
    time, S2 = results[Phi_in_val]
    print(f"  Phi_in = {Phi_in_val*1e9:.2f} nM/s → [S2] = {S2[-1]*1e9:.1f} nM")