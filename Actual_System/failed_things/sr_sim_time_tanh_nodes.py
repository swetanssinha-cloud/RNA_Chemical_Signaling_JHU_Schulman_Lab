import time as simtime
import numpy as np

from sender_receiver_tanh_nodes import (
    SenderReceiverParams,
    apply_preset,
    build_geometry,
    initialize_variables,
    build_equations,
    clip_nonnegative,
    mean_in_mask,
    simulate_sender_receiver, 
    smooth_circle_profile, 
    double_peak_diffusion
)

def main():
    """Run timing trials for the simulation."""
    
    # Use default parameters
    params = SenderReceiverParams()
    
    # Number of trials
    num_trials = 10
    trial_times = []
    
    print("Starting timing trials...")
    print(f"Running {num_trials} trials with default parameters")
    print("-" * 60)
    
    # Run trials
    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}...", end=" ", flush=True)
        
        start_time = simtime.perf_counter()
        result = simulate_sender_receiver(params, verbose=False)
        end_time = simtime.perf_counter()
        
        elapsed = end_time - start_time
        trial_times.append(elapsed)
        print(f"completed in {elapsed:.2f} seconds")
    
    # Calculate statistics
    trial_times = np.array(trial_times)
    mean_time = np.mean(trial_times)
    std_time = np.std(trial_times)
    min_time = np.min(trial_times)
    max_time = np.max(trial_times)
    
    # Report results
    print("-" * 60)
    print("Timing Results:")
    print(f"  Average time: {mean_time:.2f} ± {std_time:.2f} seconds")
    print(f"  Min time:     {min_time:.2f} seconds")
    print(f"  Max time:     {max_time:.2f} seconds")
    print(f"  Total time:   {np.sum(trial_times):.2f} seconds")
    
if __name__ == "__main__":
    main()