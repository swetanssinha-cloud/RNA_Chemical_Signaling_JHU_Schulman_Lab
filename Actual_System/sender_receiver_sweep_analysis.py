"""
Analysis script for sender-receiver reaction-diffusion parameter sweeps.

This script automatically detects which parameter was swept and generates
time-series plots and metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================

# Depletion thresholds (as fractions of initial value)
DEPLETION_THRESHOLDS = {
    't50': 0.50,  # 50% depletion
    't90': 0.10,  # 90% depletion
    't99': 0.01,  # 99% depletion
}

# Input directory (where sweep results are saved)
INPUT_DIR = Path("sweep_results")

# Plot settings
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIG_SIZE = (10, 6)
SAVE_DPI = 300
# ============================================================================


def detect_sweep_parameter(ss_data):
    """Automatically detect which parameter was swept"""
    # Look for columns that vary across runs
    possible_params = [
        'node_length_um', 'center_distance_um', 'bath_margin_um', 'dx_um',
        'total_hours', 'dt_s', 'd_gel_um2_s', 'd_solution_um2_s',
        'k_p_s_inv', 'k_d_ds_s_inv', 'k_d_ss_s_inv',
        'k_slow_M_inv_s_inv', 'k_fast_M_inv_s_inv',
        'sender_switch_nM', 'receiver_switch_nM', 'threshold_uM',
        'transition_sharpness'
    ]
    
    for param in possible_params:
        if param in ss_data.columns:
            if ss_data[param].nunique() > 1:
                return param
    
    raise ValueError("Could not detect sweep parameter. No varying parameter found in steady_state_values.csv")


def get_param_label(param_name):
    """Get human-readable label for parameter"""
    labels = {
        'node_length_um': r'Node length ($\mu$m)',
        'center_distance_um': r'Center distance ($\mu$m)',
        'bath_margin_um': r'Bath margin ($\mu$m)',
        'dx_um': r'Grid spacing ($\mu$m)',
        'total_hours': r'Simulation time (hours)',
        'dt_s': r'Time step (s)',
        'd_gel_um2_s': r'$D_{gel}$ ($\mu$m$^2$/s)',
        'd_solution_um2_s': r'$D_{solution}$ ($\mu$m$^2$/s)',
        'k_p_s_inv': r'$k_p$ (s$^{-1}$)',
        'k_d_ds_s_inv': r'$k_{d,ds}$ (s$^{-1}$)',
        'k_d_ss_s_inv': r'$k_{d,ss}$ (s$^{-1}$)',
        'k_slow_M_inv_s_inv': r'$k_{slow}$ (M$^{-1}$s$^{-1}$)',
        'k_fast_M_inv_s_inv': r'$k_{fast}$ (M$^{-1}$s$^{-1}$)',
        'sender_switch_nM': r'Sender I1O2$_0$ (nM)',
        'receiver_switch_nM': r'Receiver I2$_0$ (nM)',
        'threshold_uM': r'Threshold Th2$_0$ ($\mu$M)',
        'transition_sharpness': r'Transition sharpness',
    }
    return labels.get(param_name, param_name)


def get_param_units_conversion(param_name):
    """Get conversion factor to convenient units"""
    conversions = {
        'node_length_um': 1.0,
        'center_distance_um': 1.0,
        'bath_margin_um': 1.0,
        'dx_um': 1.0,
        'total_hours': 1.0,
        'dt_s': 1.0,
        'd_gel_um2_s': 1.0,
        'd_solution_um2_s': 1.0,
        'k_p_s_inv': 1.0,
        'k_d_ds_s_inv': 1.0,
        'k_d_ss_s_inv': 1.0,
        'k_slow_M_inv_s_inv': 1.0,
        'k_fast_M_inv_s_inv': 1.0,
        'sender_switch_nM': 1.0,
        'receiver_switch_nM': 1.0,
        'threshold_uM': 1.0,
        'transition_sharpness': 1.0,
    }
    return conversions.get(param_name, 1.0)


def find_crossing_time(time_s, values, threshold):
    """Find time when values cross threshold using linear interpolation"""
    idx = np.where(values <= threshold)[0]
    if len(idx) == 0:
        return None
    
    i = idx[0]
    if i == 0:
        return time_s[0]
    
    y1, y2 = values[i-1], values[i]
    t1, t2 = time_s[i-1], time_s[i]
    
    if y2 == y1:
        return t1
    
    t_cross = t1 + (t2 - t1) * (threshold - y1) / (y2 - y1)
    return t_cross


def analyze_time_courses(ss_data, sweep_param, sweep_values):
    """Calculate depletion times for each simulation"""
    results = []
    
    for idx, row in ss_data.iterrows():
        csv_file = row['csv_file']
        sweep_val = row[sweep_param]
        
        try:
            df = pd.read_csv(csv_file)
            time_s = df['time_s'].values
            I2_M = df['I2_M'].values
            
            # Get initial and final I2 values
            I2_initial = I2_M[0]
            I2_final = I2_M[-1]
            
            result = {
                'sweep_value': sweep_val,
                'I2_initial': I2_initial,
                'I2_final': I2_final
            }
            
            # Calculate crossing times for standard thresholds (% of initial)
            for threshold_name, threshold_frac in DEPLETION_THRESHOLDS.items():
                threshold_val = threshold_frac * I2_initial
                t_cross = find_crossing_time(time_s, I2_M, threshold_val)
                result[threshold_name] = t_cross
            
            # Calculate t50_initial (50% of initial - same as t50 above)
            result['t50_initial'] = result['t50']
            
            # Calculate t50_offrange (50% of the off-fraction range)
            # = final + 0.5*(initial - final)
            target_off_50 = I2_final + 0.5 * (I2_initial - I2_final)
            t50_offrange = find_crossing_time(time_s, I2_M, target_off_50)
            result['t50_offrange'] = t50_offrange
            
            results.append(result)
            
        except Exception as e:
            print(f"Warning: Could not process {csv_file}: {e}")
            continue
    
    return pd.DataFrame(results)


def create_plots(ss_data, depletion_data, sweep_param, sweep_values, output_dir):
    """Generate all analysis plots"""
    
    plt.style.use(PLOT_STYLE)
    
    param_label = get_param_label(sweep_param)
    param_conversion = get_param_units_conversion(sweep_param)
    
    # Convert sweep values to convenient units
    sweep_values_display = sweep_values * param_conversion
    
    # ---- Plot 1: t50_initial vs sweep parameter ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    valid = ~pd.isna(depletion_data['t50_initial'])
    ax.plot(
        depletion_data.loc[valid, 'sweep_value'] * param_conversion,
        depletion_data.loc[valid, 't50_initial'] / 3600,
        'o-', linewidth=2, markersize=8
    )
    
    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel('Time to 50% of initial [I2] (hours)', fontsize=12)
    ax.set_title(f'Time for [I2] to drop to 50% of initial value vs {param_label}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f't50_initial_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
    print(f"Saved: t50_initial_vs_{sweep_param}.png")
    plt.close()
    
    # ---- Plot 2: t50_offrange vs sweep parameter ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    valid = ~pd.isna(depletion_data['t50_offrange'])
    ax.plot(
        depletion_data.loc[valid, 'sweep_value'] * param_conversion,
        depletion_data.loc[valid, 't50_offrange'] / 3600,
        'o-', linewidth=2, markersize=8, color='green'
    )
    
    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel('Time to 50% of off-fraction [I2] (hours)', fontsize=12)
    ax.set_title(f'Time for [I2] to reach midpoint between initial and final values vs {param_label}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f't50_offrange_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
    print(f"Saved: t50_offrange_vs_{sweep_param}.png")
    plt.close()
    
    # ---- Plot 3: I2 time courses ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(sweep_values)))
    
    for idx, (sweep_val, color) in enumerate(zip(sweep_values, colors)):
        row = ss_data[ss_data[sweep_param] == sweep_val].iloc[0]
        csv_file = row['csv_file']
        
        try:
            df = pd.read_csv(csv_file)
            time_h = df['time_s'].values / 3600
            I2_nM = df['I2_M'].values * 1e9
            
            label_val = sweep_val * param_conversion
            param_name = param_label.split('(')[0].strip().replace('$', '').replace('\\', '')
            ax.plot(time_h, I2_nM, color=color, linewidth=2, 
                   label=f'{param_name} = {label_val:.2g}')
        except Exception as e:
            print(f"Warning: Could not plot {csv_file}: {e}")
            continue
    
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('[I2] (nM)', fontsize=12)
    ax.set_title(f'Receiver I2 vs time for different {param_label}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'I2_timecourses_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
    print(f"Saved: I2_timecourses_vs_{sweep_param}.png")
    plt.close()
    
    # ---- Plot 4: Th2 time courses ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    for idx, (sweep_val, color) in enumerate(zip(sweep_values, colors)):
        row = ss_data[ss_data[sweep_param] == sweep_val].iloc[0]
        csv_file = row['csv_file']
        
        try:
            df = pd.read_csv(csv_file)
            time_h = df['time_s'].values / 3600
            Th2_nM = df['Th2_M'].values * 1e9
            
            label_val = sweep_val * param_conversion
            param_name = param_label.split('(')[0].strip().replace('$', '').replace('\\', '')
            ax.plot(time_h, Th2_nM, color=color, linewidth=2, 
                   label=f'{param_name} = {label_val:.2g}')
        except Exception as e:
            print(f"Warning: Could not plot {csv_file}: {e}")
            continue
    
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('[Th2] (nM)', fontsize=12)
    ax.set_title(f'Receiver Th2 vs time for different {param_label}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'Th2_timecourses_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
    print(f"Saved: Th2_timecourses_vs_{sweep_param}.png")
    plt.close()
    
    # ---- Plot 5: S2 time courses ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    for idx, (sweep_val, color) in enumerate(zip(sweep_values, colors)):
        row = ss_data[ss_data[sweep_param] == sweep_val].iloc[0]
        csv_file = row['csv_file']
        
        try:
            df = pd.read_csv(csv_file)
            time_h = df['time_s'].values / 3600
            S2_nM = df['S2_M'].values * 1e9
            
            label_val = sweep_val * param_conversion
            param_name = param_label.split('(')[0].strip().replace('$', '').replace('\\', '')
            ax.plot(time_h, S2_nM, color=color, linewidth=2, 
                   label=f'{param_name} = {label_val:.2g}')
        except Exception as e:
            print(f"Warning: Could not plot {csv_file}: {e}")
            continue
    
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('[S2] (nM)', fontsize=12)
    ax.set_title(f'Receiver S2 vs time for different {param_label}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'S2_timecourses_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
    print(f"Saved: S2_timecourses_vs_{sweep_param}.png")
    plt.close()
    
    # ---- Plot 6: Total RNA time courses ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    for idx, (sweep_val, color) in enumerate(zip(sweep_values, colors)):
        row = ss_data[ss_data[sweep_param] == sweep_val].iloc[0]
        csv_file = row['csv_file']
        
        try:
            df = pd.read_csv(csv_file)
            time_h = df['time_s'].values / 3600
            total_RNA_nM = (df['S2_M'].values + df['S2_I2_M'].values + df['S2_Th2_M'].values) * 1e9
            
            label_val = sweep_val * param_conversion
            param_name = param_label.split('(')[0].strip().replace('$', '').replace('\\', '')
            ax.plot(time_h, total_RNA_nM, color=color, linewidth=2, 
                   label=f'{param_name} = {label_val:.2g}')
        except Exception as e:
            print(f"Warning: Could not plot {csv_file}: {e}")
            continue
    
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Total RNA (nM)', fontsize=12)
    ax.set_title(f'Receiver total RNA vs time for different {param_label}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'TotalRNA_timecourses_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
    print(f"Saved: TotalRNA_timecourses_vs_{sweep_param}.png")
    plt.close()
    
    # ---- Plot 7: Final I2 concentration vs sweep parameter ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    final_I2_nM = ss_data['final_I2_M'].values * 1e9
    
    ax.plot(sweep_values_display, final_I2_nM, 'o-', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel('Final [I2] (nM)', fontsize=12)
    ax.set_title(f'Steady-state [I2] vs {param_label}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'final_I2_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
    print(f"Saved: final_I2_vs_{sweep_param}.png")
    plt.close()
    
    # ---- Plot 8: All depletion thresholds ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    for threshold_name in DEPLETION_THRESHOLDS.keys():
        valid = ~pd.isna(depletion_data[threshold_name])
        if valid.sum() > 0:
            ax.plot(
                depletion_data.loc[valid, 'sweep_value'] * param_conversion,
                depletion_data.loc[valid, threshold_name] / 3600,
                'o-', label=threshold_name, linewidth=2, markersize=8
            )
    
    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel('Depletion Time (hours)', fontsize=12)
    ax.set_title(f'I2 Depletion Times vs {param_label}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'depletion_times_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
    print(f"Saved: depletion_times_vs_{sweep_param}.png")
    plt.close()


def print_summary(ss_data, depletion_data, sweep_param):
    """Print summary statistics"""
    param_label = get_param_label(sweep_param)
    param_conversion = get_param_units_conversion(sweep_param)
    
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nSwept parameter: {param_label}")
    print(f"Number of simulations: {len(ss_data)}")
    print(f"Converged: {ss_data['converged'].sum()} / {len(ss_data)}")
    
    print(f"\nDepletion times (hours):")
    print("-" * 80)
    for idx, row in depletion_data.iterrows():
        sweep_val = row['sweep_value'] * param_conversion
        I2_initial_nM = row['I2_initial'] * 1e9
        I2_final_nM = row['I2_final'] * 1e9
        
        param_name = param_label.split('(')[0].strip().replace('$', '').replace('\\', '')
        print(f"\n{param_name} = {sweep_val:.2g}:")
        print(f"  I2_initial = {I2_initial_nM:.2f} nM")
        print(f"  I2_final = {I2_final_nM:.4f} nM")
        
        # Standard thresholds
        for threshold_name in DEPLETION_THRESHOLDS.keys():
            t_val = row[threshold_name]
            if t_val is not None and not np.isnan(t_val):
                print(f"  {threshold_name}: {t_val/3600:.3f} hours")
            else:
                print(f"  {threshold_name}: Not reached")
        
        # Off-range threshold
        t_off = row['t50_offrange']
        if t_off is not None and not np.isnan(t_off):
            print(f"  t50_offrange: {t_off/3600:.3f} hours")
        else:
            print(f"  t50_offrange: Not reached")


def save_results_csv(depletion_data, sweep_param, output_dir):
    """Save depletion analysis results to CSV"""
    param_conversion = get_param_units_conversion(sweep_param)
    
    # Create output dataframe
    output = depletion_data.copy()
    
    # Keep sweep parameter as is (already in correct units)
    output[f'{sweep_param}_value'] = output['sweep_value']
    output = output.drop(columns=['sweep_value'])
    
    # Convert concentrations to nM
    output['I2_initial_nM'] = output['I2_initial'] * 1e9
    output['I2_final_nM'] = output['I2_final'] * 1e9
    output = output.drop(columns=['I2_initial', 'I2_final'])
    
    # Convert times to hours
    time_cols = ['t50', 't90', 't99', 't50_initial', 't50_offrange']
    for col in time_cols:
        if col in output.columns:
            output[f'{col}_hours'] = output[col] / 3600
            output = output.drop(columns=[col])
    
    output_path = output_dir / 'summary_with_thresholds.csv'
    output.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    # Load data
    ss_data = pd.read_csv(INPUT_DIR / 'steady_state_values.csv')
    
    # Detect sweep parameter
    sweep_param = detect_sweep_parameter(ss_data)
    print(f"Detected sweep parameter: {sweep_param}")
    
    # Get sweep values
    sweep_values = ss_data[sweep_param].values
    
    # Analyze time courses
    depletion_data = analyze_time_courses(ss_data, sweep_param, sweep_values)
    
    # Save results
    save_results_csv(depletion_data, sweep_param, INPUT_DIR)
    
    # Create plots
    create_plots(ss_data, depletion_data, sweep_param, sweep_values, INPUT_DIR)
    
    # Print summary
    print_summary(ss_data, depletion_data, sweep_param)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated plots:")
    print(f"  1. t50_initial_vs_{sweep_param}.png")
    print(f"  2. t50_offrange_vs_{sweep_param}.png")
    print(f"  3. I2_timecourses_vs_{sweep_param}.png")
    print(f"  4. Th2_timecourses_vs_{sweep_param}.png")
    print(f"  5. S2_timecourses_vs_{sweep_param}.png")
    print(f"  6. TotalRNA_timecourses_vs_{sweep_param}.png")
    print(f"  7. final_I2_vs_{sweep_param}.png")
    print(f"  8. depletion_times_vs_{sweep_param}.png")