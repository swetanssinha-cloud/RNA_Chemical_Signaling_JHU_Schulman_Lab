import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================
# This script automatically detects which parameter was swept based on the
# steady_state_values.csv file and generates appropriate plots

# Depletion thresholds (as fractions of initial value)
DEPLETION_THRESHOLDS = {
    't50': 0.50,  # 50% depletion
    't90': 0.10,  # 90% depletion
    't99': 0.01,  # 99% depletion
}

# Plot settings
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIG_SIZE = (10, 6)
SAVE_DPI = 300
# ============================================================================

def detect_sweep_parameter(ss_data):
    """Automatically detect which parameter was swept"""
    # Look for columns that vary across runs
    possible_params = ['Phi_in', 'I2_0', 'Th2_0', 'k_slow', 'k_fast', 'k_d_ds', 'k_d_ss']
    
    for param in possible_params:
        if param in ss_data.columns:
            if ss_data[param].nunique() > 1:
                return param
    
    raise ValueError("Could not detect sweep parameter. No varying parameter found in steady_state_values.csv")

def get_param_label(param_name):
    """Get human-readable label for parameter"""
    labels = {
        'Phi_in': r'$\Phi_{in}$ (nM/s)',
        'I2_0': r'$I2_0$ (nM)',
        'Th2_0': r'$Th2_0$ (nM)',
        'k_slow': r'$k_{slow}$ (M$^{-1}$s$^{-1}$)',
        'k_fast': r'$k_{fast}$ (M$^{-1}$s$^{-1}$)',
        'k_d_ds': r'$k_{d,ds}$ (s$^{-1}$)',
        'k_d_ss': r'$k_{d,ss}$ (s$^{-1}$)',
    }
    return labels.get(param_name, param_name)

def get_param_units_conversion(param_name):
    """Get conversion factor to convenient units"""
    conversions = {
        'Phi_in': 1e9,      # M/s to nM/s
        'I2_0': 1e9,        # M to nM
        'Th2_0': 1e9,       # M to nM
        'k_slow': 1.0,      # Keep as is
        'k_fast': 1.0,      # Keep as is
        'k_d_ds': 1.0,      # Keep as is
        'k_d_ss': 1.0,      # Keep as is
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

def create_plots(ss_data, depletion_data, sweep_param, sweep_values):
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
    plt.savefig(f't50_initial_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
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
    plt.savefig(f't50_offrange_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
    print(f"Saved: t50_offrange_vs_{sweep_param}.png")
    plt.close()
    
    # ---- Plot 3: Product (t50 * sweep_param) vs sweep parameter ----
    # Only for parameters where this makes physical sense
    if sweep_param in ['Phi_in']:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        
        valid = ~pd.isna(depletion_data['t50_initial'])
        product = (depletion_data.loc[valid, 't50_initial'].values * 
                   depletion_data.loc[valid, 'sweep_value'].values * param_conversion)
        
        ax.plot(
            depletion_data.loc[valid, 'sweep_value'] * param_conversion,
            product,
            'o-', linewidth=2, markersize=8
        )
        
        ax.set_xlabel(param_label, fontsize=12)
        ax.set_ylabel(f't50 × {param_label}', fontsize=12)
        ax.set_title(f'Product of t50 and {param_label} (to check proportionality)', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f't50_product_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
        print(f"Saved: t50_product_vs_{sweep_param}.png")
        plt.close()
    
    # ---- Plot 4: I2 time courses ----
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
            param_name = param_label.split('(')[0].strip()
            ax.plot(time_h, I2_nM, color=color, linewidth=2, 
                   label=f'{param_name} = {label_val:.2f}')
        except Exception as e:
            print(f"Warning: Could not plot {csv_file}: {e}")
            continue
    
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('[I2] (nM)', fontsize=12)
    ax.set_title(f'I2 vs time for different {param_label}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'I2_timecourses_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
    print(f"Saved: I2_timecourses_vs_{sweep_param}.png")
    plt.close()
    
    # ---- Plot 5: Final I2 concentration vs sweep parameter ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    final_I2_nM = ss_data['final_I2_M'].values * 1e9
    
    ax.plot(sweep_values_display, final_I2_nM, 'o-', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel('Final [I2] (nM)', fontsize=12)
    ax.set_title(f'Steady-state [I2] vs {param_label}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'final_I2_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
    print(f"Saved: final_I2_vs_{sweep_param}.png")
    plt.close()
    
    # ---- Plot 6: Comparison of t50, t99, and convergence time ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    valid_t50 = ~pd.isna(depletion_data['t50'])
    ax.plot(
        depletion_data.loc[valid_t50, 'sweep_value'] * param_conversion,
        depletion_data.loc[valid_t50, 't50'] / 3600,
        'o-', label='Time to 50%', linewidth=2, markersize=8
    )
    
    valid_t99 = ~pd.isna(depletion_data['t99'])
    ax.plot(
        depletion_data.loc[valid_t99, 'sweep_value'] * param_conversion,
        depletion_data.loc[valid_t99, 't99'] / 3600,
        's-', label='Time to 99% depletion', linewidth=2, markersize=8
    )
    
    # Add convergence time
    converged_mask = ss_data['converged'] == True
    if converged_mask.sum() > 0:
        ax.plot(
            ss_data.loc[converged_mask, sweep_param] * param_conversion,
            ss_data.loc[converged_mask, 'sim_time_to_ss_h'],
            '^-', label='Time to convergence', linewidth=2, markersize=8
        )
    
    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel('Time (hours)', fontsize=12)
    ax.set_title(f'Comparison of different time metrics vs {param_label}', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'time_metrics_comparison_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
    print(f"Saved: time_metrics_comparison_vs_{sweep_param}.png")
    plt.close()
    
    # ---- Plot 7: All depletion thresholds ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    for threshold_name in DEPLETION_THRESHOLDS.keys():
        valid = ~pd.isna(depletion_data[threshold_name])
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
    plt.savefig(f'depletion_times_vs_{sweep_param}.png', dpi=SAVE_DPI, bbox_inches='tight')
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
        
        print(f"\n{param_label.split('(')[0].strip()} = {sweep_val:.2f}:")
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

def save_results_csv(depletion_data, sweep_param):
    """Save depletion analysis results to CSV"""
    param_conversion = get_param_units_conversion(sweep_param)
    
    # Create output dataframe
    output = depletion_data.copy()
    
    # Convert sweep parameter to convenient units
    if sweep_param in ['Phi_in']:
        output[f'{sweep_param}_nMps'] = output['sweep_value'] * 1e9
    elif sweep_param in ['I2_0', 'Th2_0', 'S2_0']:
        output[f'{sweep_param}_nM'] = output['sweep_value'] * 1e9
    else:
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
    
    output.to_csv('summary_with_thresholds.csv', index=False)
    print(f"\nSaved: summary_with_thresholds.csv")

if __name__ == '__main__':
    # Load data
    ss_data = pd.read_csv('steady_state_values.csv')
    
    # Detect sweep parameter
    sweep_param = detect_sweep_parameter(ss_data)
    print(f"Detected sweep parameter: {sweep_param}")
    
    # Get sweep values
    sweep_values = ss_data[sweep_param].values
    
    # Analyze time courses
    depletion_data = analyze_time_courses(ss_data, sweep_param, sweep_values)
    
    # Save results
    save_results_csv(depletion_data, sweep_param)
    
    # Create plots
    create_plots(ss_data, depletion_data, sweep_param, sweep_values)
    
    # Print summary
    print_summary(ss_data, depletion_data, sweep_param)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated plots:")
    print(f"  1. t50_initial_vs_{sweep_param}.png")
    print(f"  2. t50_offrange_vs_{sweep_param}.png")
    if sweep_param in ['Phi_in']:
        print(f"  3. t50_product_vs_{sweep_param}.png")
    print(f"  4. I2_timecourses_vs_{sweep_param}.png")
    print(f"  5. final_I2_vs_{sweep_param}.png")
    print(f"  6. time_metrics_comparison_vs_{sweep_param}.png")
    print(f"  7. depletion_times_vs_{sweep_param}.png")