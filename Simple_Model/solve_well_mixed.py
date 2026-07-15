import numpy as np
import pandas as pd
import os
from fipy import CellVariable, TransientTerm, ImplicitSourceTerm, Grid1D
import time as simtime

OUTPUT_DIR = "well_mixed_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
for f in os.listdir(OUTPUT_DIR):
    if f.startswith("run_Phi_in_") and f.endswith(".csv"):
        os.remove(os.path.join(OUTPUT_DIR, f))
    if f == "summary.csv":
        os.remove(os.path.join(OUTPUT_DIR, f))

def run_simulation(Phi_in_value, params):
    """
    Run well-mixed FiPy simulation for given Phi_in value.
    Returns time, S2, I2, tw50, and S2_tot arrays.
    """
    k_slow = params['k_slow']
    k_fast = params['k_fast']
    k_d_ds = params['k_d_ds']
    k_d_ss = params['k_d_ss']
    I2_0 = (Phi_in_value / 1e-9) * params['I2_0']
    Th2_0 = params['Th2_0']
    dt_s = params['dt_s']
    n_steps = params['n_steps']
    max_sweeps = params['max_sweeps']
    tol = params['tol']

    mesh = Grid1D(nx=1, dx=1.0)

    S2 = CellVariable(name="S2", mesh=mesh, value=0.0, hasOld=True)
    I2 = CellVariable(name="I2", mesh=mesh, value=I2_0, hasOld=True)
    Th2 = CellVariable(name="Th2", mesh=mesh, value=Th2_0, hasOld=True)
    C_I2 = CellVariable(name="C_I2", mesh=mesh, value=0.0, hasOld=True)
    C_Th2 = CellVariable(name="C_Th2", mesh=mesh, value=0.0, hasOld=True)

    eq_S2 = (TransientTerm(var=S2) ==
             ImplicitSourceTerm(coeff=-k_slow * I2, var=S2) +
             ImplicitSourceTerm(coeff=-k_fast * Th2, var=S2) +
             ImplicitSourceTerm(coeff=-k_d_ss, var=S2) +
             Phi_in_value)

    eq_I2 = (TransientTerm(var=I2) == 
            k_d_ds * C_I2 +
            ImplicitSourceTerm(coeff=-k_slow * S2, var=I2))

    eq_Th2 = (TransientTerm(var=Th2) == 
            k_d_ds * C_Th2 +
            ImplicitSourceTerm(coeff=-k_fast * S2, var=Th2))

    eq_C_I2 = (TransientTerm(var=C_I2) ==
               k_slow * S2 * I2 +
               ImplicitSourceTerm(coeff=-k_d_ds, var=C_I2))

    eq_C_Th2 = (TransientTerm(var=C_Th2) ==
                k_fast * S2 * Th2 +
                ImplicitSourceTerm(coeff=-k_d_ds, var=C_Th2))

    time_history = []
    S2_history = []
    I2_history = []
    S2_tot_history = []
    tw50 = None

    for step in range(n_steps):
        S2.updateOld()
        I2.updateOld()
        Th2.updateOld()
        C_I2.updateOld()
        C_Th2.updateOld()

        for sweep in range(max_sweeps):
            res_S2 = eq_S2.sweep(dt=dt_s)
            res_I2 = eq_I2.sweep(dt=dt_s)
            res_Th2 = eq_Th2.sweep(dt=dt_s)
            res_C_I2 = eq_C_I2.sweep(dt=dt_s)
            res_C_Th2 = eq_C_Th2.sweep(dt=dt_s)

            residual = max(res_S2, res_I2, res_Th2, res_C_I2, res_C_Th2)
            if residual < tol:
                break

        current_time = (step + 1) * dt_s

        if tw50 is None and I2.value[0] <= 50e-9 and I2.old[0] > 50e-9:
            tw50 = current_time

        time_history.append(current_time)
        S2_history.append(S2.value[0])
        I2_history.append(I2.value[0])
        S2_tot_history.append(S2.value[0] + C_I2.value[0] + C_Th2.value[0])

    if tw50 is None:
        tw50 = np.nan

    return (
        np.array(time_history),
        np.array(S2_history),
        np.array(I2_history),
        tw50,
        np.array(S2_tot_history)
    )


def save_run_to_csv(Phi_in_value, time, S2, I2, S2_tot, tw50, output_dir=OUTPUT_DIR):
    """Save one simulation run to a CSV file."""
    phi_label = f"{Phi_in_value * 1e9:.3f}".replace(".", "p")
    filename = os.path.join(output_dir, f"run_Phi_in_{phi_label}nMps.csv")

    df = pd.DataFrame({
        "time_s": time,
        "S2_M": S2,
        "I2_M": I2,
        "S2_tot_M": S2_tot
    })

    df.to_csv(filename, index=False)
    return filename


if __name__ == "__main__":
    params = {
        'k_slow': 1e5,
        'k_fast': 1e6,
        'k_d_ds': 3e-4,
        'k_d_ss': 3e-4,
        'I2_0': 100e-9,
        'Th2_0': 5000e-9,
        'dt_s': 60.0,
        'n_steps': 480,
        'max_sweeps': 20,
        'tol': 1e-10
    }

    Phi_in_array = np.array([0.125,0.25, 0.5,1,2,4,5]) * 1e-9#([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5]) * 1e-9

    summary_rows = []

    start_time = simtime.perf_counter()

    for Phi_in_val in Phi_in_array:
        print(f"Running simulation for Phi_in = {Phi_in_val*1e9:.2f} nM/s...")
        time, S2, I2, tw50, S2_tot = run_simulation(Phi_in_val, params)

        csv_file = save_run_to_csv(Phi_in_val, time, S2, I2, S2_tot, tw50)

        summary_rows.append({
            "Phi_in_nMps": Phi_in_val * 1e9,
            "tw50_s": tw50,
            "final_S2_nM": S2[-1] * 1e9,
            "final_S2_tot_nM": S2_tot[-1] * 1e9,
            "final_I2_nM": I2[-1] * 1e9,
            "csv_file": csv_file
        })

        print(f"Saved {csv_file}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

    end_time = simtime.perf_counter()
    print(f"\nTotal simulation time: {end_time - start_time:.2f} seconds")
    print(f"Saved summary to {os.path.join(OUTPUT_DIR, 'summary.csv')}")