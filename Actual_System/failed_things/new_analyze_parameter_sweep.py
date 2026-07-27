"""
Analysis script for sender-receiver reaction-diffusion parameter sweep results.

This script automatically detects which parameter was swept, analyzes time courses,
calculates depletion metrics, and generates comprehensive plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================
# Input directory (should match OUTPUT_DIR from simulation script)
INPUT_DIR = Path("sweep_results")
INPUT_DIR = Path("sweep_results_fixed_domain") #just for the fixed domain case

# Depletion thresholds for I2 and Th2 (as fractions of initial value)
DEPLETION_THRESHOLDS = {
    't50': 0.50,  # 50% depletion
    't90': 0.10,  # 90% depletion  
    't99': 0.01,  # 99% depletion
}

# Response thresholds for S2 (absolute concentrations in M)
S2_RESPONSE_THRESHOLDS = {
    's2_t50_max': 0.50,  # 50% of max S2
    's2_t90_max': 0.90,  # 90% of max S2
}

# Plot settings
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIG_SIZE = (10, 6)
SAVE_DPI = 300

# ============================================================================

def detect_sweep_parameter(ss_data):
    """Automatically detect which parameter was swept"""
    # Possible parameters that might be swept
    possible_params = [
        'node_length_um', 'center_distance_um', 'bath_margin_um', 'dx_um',
        'total_hours', 'dt_s', 'd_gel_um2_s', 'd_solution_um2_s',
        'k_p_s_inv', 'k_d_ds_s_inv', 'k_d_ss_s_inv',
        'k_slow_M_inv_s_inv', 'k_fast_M_inv_s_inv',
        'sender_switch_nM', 'receiver_switch_nM',
        'threshold_uM', 'transition_sharpness'
    ]
    
    for param in possible_params:
        if param in ss_data.columns:
            if ss_data[param].nunique() > 1:
                return param
    
    raise ValueError("Could not detect sweep parameter. No varying parameter found in steady_state_values.csv")


def get_param_label(param_name):
    """Get human-readable label for parameter"""
    labels = {
        'node_length_um': r'Node Length (μm)',
        'center_distance_um': r'Center Distance (μm)',
        'bath_margin_um': r'Bath Margin (μm)',
        'dx_um': r'Grid Spacing (μm)',
        'total_hours': r'Total Time (hours)',
        'dt_s': r'Time Step (s)',
        'd_gel_um2_s': r'$D_{gel}$ (μm²/s)',
        'd_solution_um2_s': r'$D_{solution}$ (μm²/s)',
        'k_p_s_inv': r'$k_p$ (s⁻¹)',
        'k_d_ds_s_inv': r'$k_{d,ds}$ (s⁻¹)',
        'k_d_ss_s_inv': r'$k_{d,ss}$ (s⁻¹)',
        'k_slow_M_inv_s_inv': r'$k_{slow}$ (M⁻¹s⁻¹)',
        'k_fast_M_inv_s_inv': r'$k_{fast}$ (M⁻¹s⁻¹)',
        'sender_switch_nM': r'Sender Switch (nM)',
        'receiver_switch_nM': r'Receiver Switch (nM)',
        'threshold_uM': r'Threshold (μM)',
        'transition_sharpness': r'Transition Sharpness',
    }
    return labels.get(param_name, param_name)


def get_param_units_conversion(param_name):
    """Get conversion factor to convenient units for display"""
    conversions = {
        'threshold_uM': 1.0,  # Already in μM
        'sender_switch_nM': 1.0,  # Already in nM
        'receiver_switch_nM': 1.0,  # Already in nM
        # Add others as needed - most geometric params are already in convenient units
    }
    return conversions.get(param_name, 1.0)


def find_crossing_time(time_s, values, threshold, direction='down'):
    """
    Find time when values cross threshold using linear interpolation.
    
    Parameters:
    -----------
    time_s : array
        Time points
    values : array
        Values at each time point
    threshold : float
        Threshold to cross
    direction : str
        'down' for decreasing (depletion), 'up' for increasing (accumulation)
    """
    if direction == 'down':
        idx = np.where(values <= threshold)[0]
    else:  # direction == 'up'
        idx = np.where(values >= threshold)[0]
    
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


def analyze_time_courses(ss_data, sweep_param):
    """Calculate depletion and response times for each simulation"""
    results = []
    
    for idx, row in ss_data.iterrows():
        csv_file = INPUT_DIR / row['csv_file']
        sweep_val = row[sweep_param]
        
        try:
            df = pd.read_csv(csv_file)
            time_s = df['time_s'].values
            I2_M = df['I2_M'].values
            Th2_M = df['Th2_M'].values
            S2_M = df['S2_M'].values
            
            # Get initial and final values
            I2_initial = I2_M[0]
            I2_final = I2_M[-1]
            Th2_initial = Th2_M[0]
            Th2_final = Th2_M[-1]
            S2_initial = S2_M[0]
            S2_final = S2_M[-1]
            S2_max = np.max(S2_M)
            
            result = {
                'sweep_value': sweep_val,
                'I2_initial': I2_initial,
                'I2_final': I2_final,
                'Th2_initial': Th2_initial,
                'Th2_final': Th2_final,
                'S2_initial': S2_initial,
                'S2_final': S2_final,
                'S2_max': S2_max,
            }
            
            # ---- I2 Depletion Times ----
            # Calculate crossing times for standard thresholds (% of initial)
            for threshold_name, threshold_frac in DEPLETION_THRESHOLDS.items():
                threshold_val = threshold_frac * I2_initial
                t_cross = find_crossing_time(time_s, I2_M, threshold_val, direction='down')
                result[f'I2_{threshold_name}'] = t_cross
            
            # Calculate I2 t50_offrange (50% of the off-fraction range)
            target_I2_off_50 = I2_final + 0.5 * (I2_initial - I2_final)
            t50_I2_offrange = find_crossing_time(time_s, I2_M, target_I2_off_50, direction='down')
            result['I2_t50_offrange'] = t50_I2_offrange
            
            # ---- Th2 Depletion Times ----
            for threshold_name, threshold_frac in DEPLETION_THRESHOLDS.items():
                threshold_val = threshold_frac * Th2_initial
                t_cross = find_crossing_time(time_s, Th2_M, threshold_val, direction='down')
                result[f'Th2_{threshold_name}'] = t_cross
            
            # Calculate Th2 t50_offrange
            target_Th2_off_50 = Th2_final + 0.5 * (Th2_initial - Th2_final)
            t50_Th2_offrange = find_crossing_time(time_s, Th2_M, target_Th2_off_50, direction='down')
            result['Th2_t50_offrange'] = t50_Th2_offrange
            
            # ---- S2 Response Times ----
            for threshold_name, threshold_frac in S2_RESPONSE_THRESHOLDS.items():
                threshold_val = threshold_frac * S2_max
                t_cross = find_crossing_time(time_s, S2_M, threshold_val, direction='up')
                result[f'S2_{threshold_name}'] = t_cross
            
            # Time to reach 50% of final S2 (from below)
            target_S2_50_final = 0.5 * S2_final
            t_S2_50_final = find_crossing_time(time_s, S2_M, target_S2_50_final, direction='up')
            result['S2_t50_final'] = t_S2_50_final
            
            results.append(result)
            
        except Exception as e:
            print(f"Warning: Could not process {csv_file}: {e}")
            continue
    
    return pd.DataFrame(results)


def create_plots(ss_data, analysis_data, sweep_param):
    """Generate all analysis plots"""
    
    plt.style.use(PLOT_STYLE)
    
    param_label = get_param_label(sweep_param)
    param_conversion = get_param_units_conversion(sweep_param)
    
    # Convert sweep values to convenient units
    sweep_values = ss_data[sweep_param].values
    sweep_values_display = sweep_values * param_conversion
    
    # ---- Plot 1: I2 time courses ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(sweep_values)))
    
    for idx, (sweep_val, color) in enumerate(zip(sweep_values, colors)):
        row = ss_data[ss_data[sweep_param] == sweep_val].iloc[0]
        csv_file = INPUT_DIR / row['csv_file']
        
        try:
            df = pd.read_csv(csv_file)
            time_h = df['time_s'].values / 3600
            I2_nM = df['I2_M'].values * 1e9
            
            label_val = sweep_val * param_conversion
            param_name = param_label.split('(')[0].strip()
            ax.plot(time_h, I2_nM, color=color, linewidth=2, 
                   label=f'{param_name} = {label_val:.3g}')
        except Exception as e:
            print(f"Warning: Could not plot {csv_file}: {e}")
            continue
    
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('[I2] in Receiver (nM)', fontsize=12)
    ax.set_title(f'I2 Depletion Dynamics vs {param_label}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(INPUT_DIR / f'I2_timecourses_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
    print(f"Saved: I2_timecourses_vs_{sweep_param}.png")
    plt.close()
    
    # ---- Plot 2: Th2 time courses ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    for idx, (sweep_val, color) in enumerate(zip(sweep_values, colors)):
        row = ss_data[ss_data[sweep_param] == sweep_val].iloc[0]
        csv_file = INPUT_DIR / row['csv_file']
        
        try:
            df = pd.read_csv(csv_file)
            time_h = df['time_s'].values / 3600
            Th2_nM = df['Th2_M'].values * 1e9
            
            label_val = sweep_val * param_conversion
            param_name = param_label.split('(')[0].strip()
            ax.plot(time_h, Th2_nM, color=color, linewidth=2,
                   label=f'{param_name} = {label_val:.3g}')
        except Exception as e:
            print(f"Warning: Could not plot {csv_file}: {e}")
            continue
    
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('[Th2] in Receiver (nM)', fontsize=12)
    ax.set_title(f'Th2 Depletion Dynamics vs {param_label}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(INPUT_DIR / f'Th2_timecourses_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
    print(f"Saved: Th2_timecourses_vs_{sweep_param}.png")
    plt.close()
    
    # ---- Plot 3: S2 time courses ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    for idx, (sweep_val, color) in enumerate(zip(sweep_values, colors)):
        row = ss_data[ss_data[sweep_param] == sweep_val].iloc[0]
        csv_file = INPUT_DIR / row['csv_file']
        
        try:
            df = pd.read_csv(csv_file)
            time_h = df['time_s'].values / 3600
            S2_nM = df['S2_M'].values * 1e9
            
            label_val = sweep_val * param_conversion
            param_name = param_label.split('(')[0].strip()
            ax.plot(time_h, S2_nM, color=color, linewidth=2,
                   label=f'{param_name} = {label_val:.3g}')
        except Exception as e:
            print(f"Warning: Could not plot {csv_file}: {e}")
            continue
    
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('[S2] in Receiver (nM)', fontsize=12)
    ax.set_title(f'S2 Production Dynamics vs {param_label}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(INPUT_DIR / f'S2_timecourses_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
    print(f"Saved: S2_timecourses_vs_{sweep_param}.png")
    plt.close()
    
    # ---- Plot 4: I2 t50 vs sweep parameter ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    valid = ~pd.isna(analysis_data['I2_t50'])
    ax.plot(
        analysis_data.loc[valid, 'sweep_value'] * param_conversion,
        analysis_data.loc[valid, 'I2_t50'] / 3600,
        'o-', linewidth=2, markersize=8, color='blue'
    )
    
    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel('Time to 50% I2 Depletion (hours)', fontsize=12)
    ax.set_title(f'I2 Half-Life vs {param_label}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(INPUT_DIR / f'I2_t50_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
    print(f"Saved: I2_t50_vs_{sweep_param}.png")
    plt.close()
    
    # ---- Plot 5: Th2 t50 vs sweep parameter ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    valid = ~pd.isna(analysis_data['Th2_t50'])
    ax.plot(
        analysis_data.loc[valid, 'sweep_value'] * param_conversion,
        analysis_data.loc[valid, 'Th2_t50'] / 3600,
        'o-', linewidth=2, markersize=8, color='red'
    )
    
    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel('Time to 50% Th2 Depletion (hours)', fontsize=12)
    ax.set_title(f'Th2 Half-Life vs {param_label}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(INPUT_DIR / f'Th2_t50_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
    print(f"Saved: Th2_t50_vs_{sweep_param}.png")
    plt.close()
    
    # ---- Plot 6: S2 response time vs sweep parameter ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    valid = ~pd.isna(analysis_data['S2_s2_t50_max'])
    ax.plot(
        analysis_data.loc[valid, 'sweep_value'] * param_conversion,
        analysis_data.loc[valid, 'S2_s2_t50_max'] / 3600,
        'o-', linewidth=2, markersize=8, color='green'
    )
    
    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel('Time to 50% Max S2 (hours)', fontsize=12)
    ax.set_title(f'S2 Response Time vs {param_label}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(INPUT_DIR / f'S2_response_time_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
    print(f"Saved: S2_response_time_vs_{sweep_param}.png")
    plt.close()

    # Add this plot after Plot 6 in the create_plots function

# ---- Plot 7: Total S2 time courses ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for idx, (sweep_val, color) in enumerate(zip(sweep_values, colors)):
        row = ss_data[ss_data[sweep_param] == sweep_val].iloc[0]
        csv_file = row['csv_file']
        
        try:
            df = pd.read_csv(csv_file)
            time_h = df['time_s'].values / 3600
            # Total S2 = free S2 + S2:I2 + S2:Th2
            total_S2_nM = (df['S2_M'].values + df['S2_I2_M'].values + df['S2_Th2_M'].values) * 1e9
            
            label_val = sweep_val * param_conversion
            param_name = param_label.split('(')[0].strip().replace('$', '').replace('\\', '')
            ax.plot(time_h, total_S2_nM, color=color, linewidth=2, 
                label=f'{param_name} = {label_val:.2g}')
        except Exception as e:
            print(f"Warning: Could not plot {csv_file}: {e}")
            continue

    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Total S2 (nM)', fontsize=12)
    ax.set_title(f'Receiver Total S2 (free + S2:I2 + S2:Th2) vs time for different {param_label}', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(INPUT_DIR / f'TotalS2_timecourses_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
    print(f"Saved: TotalS2_timecourses_vs_{sweep_param}.png")
    plt.close()
    
    # ---- Plot 7: Final concentrations vs sweep parameter ----
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # I2 final
    final_I2_nM = ss_data['final_I2_M'].values * 1e9
    ax1.plot(sweep_values_display, final_I2_nM, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel(param_label, fontsize=12)
    ax1.set_ylabel('Final [I2] (nM)', fontsize=12)
    ax1.set_title('Steady-State I2', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Th2 final
    final_Th2_nM = ss_data['final_Th2_M'].values * 1e9
    ax2.plot(sweep_values_display, final_Th2_nM, 'o-', linewidth=2, markersize=8, color='red')
    ax2.set_xlabel(param_label, fontsize=12)
    ax2.set_ylabel('Final [Th2] (nM)', fontsize=12)
    ax2.set_title('Steady-State Th2', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # S2 final
    final_S2_nM = ss_data['final_S2_M'].values * 1e9
    ax3.plot(sweep_values_display, final_S2_nM, 'o-', linewidth=2, markersize=8, color='green')
    ax3.set_xlabel(param_label, fontsize=12)
    ax3.set_ylabel('Final [S2] (nM)', fontsize=12)
    ax3.set_title('Steady-State S2', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(INPUT_DIR / f'final_concentrations_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
    print(f"Saved: final_concentrations_vs_{sweep_param}.png")
    plt.close()
    
    # ---- Plot 8: All I2 depletion thresholds ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    for threshold_name in DEPLETION_THRESHOLDS.keys():
        col_name = f'I2_{threshold_name}'
        valid = ~pd.isna(analysis_data[col_name])
        if valid.sum() > 0:
            ax.plot(
                analysis_data.loc[valid, 'sweep_value'] * param_conversion,
                analysis_data.loc[valid, col_name] / 3600,
                'o-', label=threshold_name.replace('t', '').upper(), linewidth=2, markersize=8
            )
    
    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel('I2 Depletion Time (hours)', fontsize=12)
    ax.set_title(f'I2 Depletion Kinetics vs {param_label}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, title='Depletion Level')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(INPUT_DIR / f'I2_depletion_times_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
    print(f"Saved: I2_depletion_times_vs_{sweep_param}.png")
    plt.close()
    
    # ---- Plot 9: Comparison of convergence time vs t50 metrics ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # I2 t50
    valid_I2 = ~pd.isna(analysis_data['I2_t50'])
    ax.plot(
        analysis_data.loc[valid_I2, 'sweep_value'] * param_conversion,
        analysis_data.loc[valid_I2, 'I2_t50'] / 3600,
        'o-', label='I2 t50', linewidth=2, markersize=8
    )
    
    # Th2 t50
    valid_Th2 = ~pd.isna(analysis_data['Th2_t50'])
    ax.plot(
        analysis_data.loc[valid_Th2, 'sweep_value'] * param_conversion,
        analysis_data.loc[valid_Th2, 'Th2_t50'] / 3600,
        's-', label='Th2 t50', linewidth=2, markersize=8
    )
    
    # S2 response time
    valid_S2 = ~pd.isna(analysis_data['S2_s2_t50_max'])
    ax.plot(
        analysis_data.loc[valid_S2, 'sweep_value'] * param_conversion,
        analysis_data.loc[valid_S2, 'S2_s2_t50_max'] / 3600,
        '^-', label='S2 t50 to max', linewidth=2, markersize=8
    )
    
    # Convergence time
    converged_mask = ss_data['converged'] == True
    if converged_mask.sum() > 0:
        ax.plot(
            ss_data.loc[converged_mask, sweep_param] * param_conversion,
            ss_data.loc[converged_mask, 'sim_time_to_ss_h'],
            'd-', label='Time to convergence', linewidth=2, markersize=8
        )
    
    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel('Time (hours)', fontsize=12)
    ax.set_title(f'Time Metrics Comparison vs {param_label}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(INPUT_DIR / f'time_metrics_comparison_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
    print(f"Saved: time_metrics_comparison_vs_{sweep_param}.png")
    plt.close()
    
    # ---- Plot 10: Max S2 vs sweep parameter ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    S2_max_nM = analysis_data['S2_max'].values * 1e9
    ax.plot(sweep_values_display, S2_max_nM, 'o-', linewidth=2, markersize=8, color='darkgreen')
    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel('Max [S2] Reached (nM)', fontsize=12)
    ax.set_title(f'Peak S2 Production vs {param_label}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(INPUT_DIR / f'S2_max_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
    print(f"Saved: S2_max_vs_{sweep_param}.png")
    plt.close()


def print_summary(ss_data, analysis_data, sweep_param):
    """Print summary statistics"""
    param_label = get_param_label(sweep_param)
    param_conversion = get_param_units_conversion(sweep_param)
    
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nSwept parameter: {param_label}")
    print(f"Number of simulations: {len(ss_data)}")
    print(f"Converged: {ss_data['converged'].sum()} / {len(ss_data)}")
    
    print(f"\nDetailed Results:")
    print("-" * 80)
    
    for idx, (ss_row, analysis_row) in enumerate(zip(ss_data.iterrows(), analysis_data.iterrows())):
        ss_row = ss_row[1]  # Get the actual row data
        analysis_row = analysis_row[1]
        
        sweep_val = ss_row[sweep_param] * param_conversion
        
        print(f"\n{param_label.split('(')[0].strip()} = {sweep_val:.3g}:")
        print(f"  Converged: {ss_row['converged']}")
        print(f"  Status: {ss_row['status']}")
        print(f"  Sim time to SS: {ss_row['sim_time_to_ss_h']:.3f} hours")
        print(f"  Wall time: {ss_row['wall_time_s']:.1f} seconds")
        
        print(f"\n  Initial Concentrations:")
        print(f"    I2:  {analysis_row['I2_initial']*1e9:.2f} nM")
        print(f"    Th2: {analysis_row['Th2_initial']*1e9:.2f} nM")
        print(f"    S2:  {analysis_row['S2_initial']*1e9:.2f} nM")
        
        print(f"\n  Final Concentrations:")
        print(f"    I2:  {analysis_row['I2_final']*1e9:.4f} nM")
        print(f"    Th2: {analysis_row['Th2_final']*1e9:.4f} nM")
        print(f"    S2:  {analysis_row['S2_final']*1e9:.2f} nM")
        print(f"    S2 max: {analysis_row['S2_max']*1e9:.2f} nM")
        
        print(f"\n  I2 Depletion Times:")
        for threshold_name in DEPLETION_THRESHOLDS.keys():
            col_name = f'I2_{threshold_name}'
            t_val = analysis_row[col_name]
            if t_val is not None and not np.isnan(t_val):
                print(f"    {threshold_name}: {t_val/3600:.3f} hours")
            else:
                print(f"    {threshold_name}: Not reached")
        
        print(f"\n  Th2 Depletion Times:")
        for threshold_name in DEPLETION_THRESHOLDS.keys():
            col_name = f'Th2_{threshold_name}'
            t_val = analysis_row[col_name]
            if t_val is not None and not np.isnan(t_val):
                print(f"    {threshold_name}: {t_val/3600:.3f} hours")
            else:
                print(f"    {threshold_name}: Not reached")
        
        print(f"\n  S2 Response Times:")
        for threshold_name in S2_RESPONSE_THRESHOLDS.keys():
            col_name = f'S2_{threshold_name}'
            t_val = analysis_row[col_name]
            if t_val is not None and not np.isnan(t_val):
                print(f"    {threshold_name}: {t_val/3600:.3f} hours")
            else:
                print(f"    {threshold_name}: Not reached")


def save_results_csv(analysis_data, sweep_param):
    """Save analysis results to CSV"""
    param_conversion = get_param_units_conversion(sweep_param)
    
    # Create output dataframe
    output = analysis_data.copy()
    
    # Convert sweep parameter display name
    output[f'{sweep_param}_value'] = output['sweep_value']
    output = output.drop(columns=['sweep_value'])
    
    # Convert concentrations to nM
    conc_cols = ['I2_initial', 'I2_final', 'Th2_initial', 'Th2_final', 
                 'S2_initial', 'S2_final', 'S2_max']
    for col in conc_cols:
        if col in output.columns:
            output[f'{col}_nM'] = output[col] * 1e9
            output = output.drop(columns=[col])
    
    # Convert times to hours
    time_cols = [col for col in output.columns if col.startswith(('I2_t', 'Th2_t', 'S2_t'))]
    for col in time_cols:
        if col in output.columns:
            output[f'{col}_hours'] = output[col] / 3600
            output = output.drop(columns=[col])
    
    output_path = INPUT_DIR / 'analysis_summary.csv'
    output.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    print("="*80)
    print("SENDER-RECEIVER PARAMETER SWEEP ANALYSIS")
    print("="*80)
    
    # Load data
    ss_path = INPUT_DIR / 'steady_state_values.csv'
    if not ss_path.exists():
        raise FileNotFoundError(f"Could not find {ss_path}. Run the simulation script first.")
    
    ss_data = pd.read_csv(ss_path)
    print(f"\nLoaded data from: {ss_path}")
    
    # Detect sweep parameter
    sweep_param = detect_sweep_parameter(ss_data)
    print(f"Detected sweep parameter: {sweep_param}")
    print(f"  Display label: {get_param_label(sweep_param)}")
    
    # Analyze time courses
    print("\nAnalyzing time courses...")
    analysis_data = analyze_time_courses(ss_data, sweep_param)
    
    # Save results
    save_results_csv(analysis_data, sweep_param)
    
    # Create plots
    print("\nGenerating plots...")
    create_plots(ss_data, analysis_data, sweep_param)
    
    # Print summary
    print_summary(ss_data, analysis_data, sweep_param)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated plots:")
    print(f"  1. I2_timecourses_vs_{sweep_param}.png")
    print(f"  2. Th2_timecourses_vs_{sweep_param}.png")
    print(f"  3. S2_timecourses_vs_{sweep_param}.png")
    print(f"  4. I2_t50_vs_{sweep_param}.png")
    print(f"  5. Th2_t50_vs_{sweep_param}.png")
    print(f"  6. S2_response_time_vs_{sweep_param}.png")
    print(f"  7. final_concentrations_vs_{sweep_param}.png")
    print(f"  8. I2_depletion_times_vs_{sweep_param}.png")
    print(f"  9. time_metrics_comparison_vs_{sweep_param}.png")
    print(f" 10. S2_max_vs_{sweep_param}.png")
    print(f"\nAll results saved to: {INPUT_DIR}")