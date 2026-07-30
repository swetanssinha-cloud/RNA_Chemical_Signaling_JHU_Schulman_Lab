"""
2D Tethered Genelet Model - With Adaptive Mesh Refinement
"""

import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, ImplicitSourceTerm
from fipy.tools import numerix
import csv
import pandas as pd

from Functions import calculate_total_amount

# =============================================================================
# PARAMETERS (same as before)
# =============================================================================

D_solution = 150.0 
D_gel = 60.0
k_p = 0.2 #1/s
k_d_ds = 3e-4 #1/s
k_d_ss = 3e-4 #1/s
k_slow = 1e5 * 1e-6 # 1/(Ms) * microMolar
k_fast = 1e6 * 1e-6 # 1/(Ms) * microMolar
 
I1O2_init = 0.1 #(in uM) - 100 nM
I2_init = 0.1 #(in uM) - 100 nM
Th2_init = 5.0 #(in uM) - 5000 nM

node_size = 50.0
node_diameter = 75
node_radius = node_diameter / 2
bath_margin = 250
distance_between = 200 # Test at large distance
total_width = 1e4 #10000 μm = 1 cm
total_height = 1e3 #1000 μm = 1 mm

dt = 30.0
total_time = 8 * 3600
n_steps = int(total_time / dt)
save_interval_time = 60.0
save_interval_steps = int(save_interval_time / dt)

def smooth_circular_profile(x, y, center_x, center_y, radius, 
                            value_inside, value_outside, 
                            transition_width=10.0):
    """
    Create smooth circular concentration/diffusion profile using hyperbolic tangent.
    
    This replaces sharp boolean masks with smooth transitions to eliminate
    divide-by-zero errors in gradient calculations.
    
    Parameters:
    -----------
    x, y : numpy arrays
        Cell center coordinates from mesh.cellCenters (μm)
    center_x, center_y : float
        Center of circular node (μm)
    radius : float
        Radius of circular node (μm)
    value_inside : float
        Value at center of node (e.g., I2_init=0.1 μM or D_gel=60.0 μm²/s)
    value_outside : float
        Value far from node (e.g., 0.0 μM or D_solution=150.0 μm²/s)
    transition_width : float
        Width of smooth transition region (μm)
        Recommended: 2-5× the finest mesh spacing
        Smaller = sharper transition (more like boolean mask)
        Larger = smoother transition (more gradual)
    
    Returns:
    --------
    profile : numpy array
        Smooth profile values at each cell center
    
    Mathematical Form:
    ------------------
    profile(r) = U + (H/2) * [tanh(c*(R - r)) + 1]
    
    where:
        r = distance from center = sqrt((x-h)² + (y-k)²)
        R = radius
        H = value_inside - value_outside (height)
        U = value_outside (baseline)
        c = 1/transition_width (steepness parameter)
    
    At r=0 (center):      profile ≈ value_inside
    At r=R (boundary):    profile ≈ (value_inside + value_outside)/2
    At r→∞ (far away):    profile ≈ value_outside
    """
    # Calculate distance from center
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Steepness parameter (larger c = sharper transition)
    c = 1.0 / transition_width
    
    # Height and baseline
    H = value_inside - value_outside
    U = value_outside
    
    # Hyperbolic tangent profile
    # tanh(c*(R-r)) varies from +1 at r=0 to -1 as r→∞
    profile = U + (H / 2.0) * (np.tanh(c * (radius - distance)) + 1.0)
    
    return profile

# =============================================================================
# ADAPTIVE MESH GENERATION FUNCTIONS
# =============================================================================

def create_adaptive_mesh_for_simulation(
    node_size=50.0,
    sender_center=None,
    receiver_center=None,
    fine_dx=5.0,
    coarse_dx=40.0,
    box_padding=200.0,
    transition_width=100.0,
    total_width=1e4,
    total_height=1e3
):
    """
    Create adaptive mesh for the 2D genelet simulation.
    
    Parameters:
    -----------
    node_size : float
        Size of sender/receiver nodes (μm)
    sender_center : float
        X-coordinate of sender center (μm)
    receiver_center : float
        X-coordinate of receiver center (μm)
    fine_dx : float
        Fine mesh spacing near nodes (μm)
    coarse_dx : float
        Coarse mesh spacing away from nodes (μm)
    box_padding : float
        Padding around nodes to define refinement box (μm)
    transition_width : float
        Width of transition between fine and coarse (μm)
    total_width : float
        Total domain width (μm)
    total_height : float
        Total domain height (μm)
    """
    
    # Calculate sender and receiver positions if not provided
    if sender_center is None:
        sender_center = total_width / 2 - distance_between / 2
    if receiver_center is None:
        receiver_center = total_width / 2 + distance_between / 2
    
    # Calculate bounding box that encompasses both nodes
    node_centers_x = [sender_center, receiver_center]
    node_centers_y = [total_height / 2]  # Center vertically
    
    # Refinement box boundaries in X
    x_min_nodes = min(node_centers_x) - node_size/2
    x_max_nodes = max(node_centers_x) + node_size/2
    refinement_x_min = x_min_nodes - box_padding
    refinement_x_max = x_max_nodes + box_padding
    
    # Refinement box boundaries in Y
    y_center = node_centers_y[0]
    refinement_y_min = y_center - node_size/2 - box_padding
    refinement_y_max = y_center + node_size/2 + box_padding
    
    def distance_to_box(x, y, x_min, x_max, y_min, y_max):
        """
        Calculate the minimum distance from point (x,y) to the box boundary.
        Returns 0 if inside the box, positive distance if outside.
        
        """
        # Correct way: calculate each distance component
        dx = np.maximum(np.maximum(x_min - x, 0), x - x_max)
        dy = np.maximum(np.maximum(y_min - y, 0), y - y_max)
        return np.sqrt(dx**2 + dy**2)


    def calculate_refinement_factor(x, y):
        """
        Calculate refinement factor based on distance to refinement box.
        Returns value between 0 (fine) and 1 (coarse).
        """
        dist = distance_to_box(x, y, refinement_x_min, refinement_x_max,
                              refinement_y_min, refinement_y_max)
        
        if dist < transition_width:
            # Smooth transition using tanh
            blend = 0.5 * (1 + np.tanh(
                (dist - transition_width/2) / (transition_width/10)
            ))
        else:
            blend = 1.0
        
        return blend
    
    def create_adaptive_spacing_1D(total_length, positions_other_dim,
                                    fine_dx, coarse_dx, is_x_direction=True):
        """
        Create 1D spacing array that depends on position in both dimensions.
        """
        positions = [0.0]
        current_pos = 0.0
        
        while current_pos < total_length:
            # Sample refinement at current position across other dimension
            refinement_samples = []
            sample_step = max(1, len(positions_other_dim)//20)
            
            for other_pos in positions_other_dim[::sample_step]:
                if is_x_direction:
                    x, y = current_pos, other_pos
                else:
                    x, y = other_pos, current_pos
                
                blend = calculate_refinement_factor(x, y)
                refinement_samples.append(blend)
            
            # Use minimum blend (finest mesh needed along this line)
            blend = min(refinement_samples) if refinement_samples else 1.0
            dx_local = fine_dx + (coarse_dx - fine_dx) * blend
            
            current_pos += dx_local
            if current_pos < total_length:
                positions.append(current_pos)
        
        # Ensure we end at total_length
        if positions[-1] < total_length:
            positions.append(total_length)
        
        positions = np.array(positions)
        dx_array = np.diff(positions)
        return positions, dx_array
    
    # First pass: create preliminary Y spacing
    y_positions_prelim = np.linspace(0, total_height, 100)
    
    # Create X spacing considering all Y positions
    x_positions, dx_array = create_adaptive_spacing_1D(
        total_width, y_positions_prelim,
        fine_dx, coarse_dx, is_x_direction=True
    )
    
    # Create Y spacing considering all X positions
    y_positions, dy_array = create_adaptive_spacing_1D(
        total_height, x_positions,
        fine_dx, coarse_dx, is_x_direction=False
    )
    
    # Create the mesh
    mesh = Grid2D(dx=dx_array, dy=dy_array)
    
    # Calculate refinement box dimensions
    box_width = refinement_x_max - refinement_x_min
    box_height = refinement_y_max - refinement_y_min
    
    print(f"\nAdaptive Mesh Created:")
    print(f"  Total cells: {mesh.numberOfCells}")
    print(f"  X cells: {len(dx_array)}")
    print(f"  Y cells: {len(dy_array)}")
    print(f"  Min dx: {dx_array.min():.2f} μm")
    print(f"  Max dx: {dx_array.max():.2f} μm")
    print(f"  Min dy: {dy_array.min():.2f} μm")
    print(f"  Max dy: {dy_array.max():.2f} μm")
    print(f"\nRefinement box:")
    print(f"  X: [{refinement_x_min:.1f}, {refinement_x_max:.1f}] μm (width: {box_width:.1f} μm)")
    print(f"  Y: [{refinement_y_min:.1f}, {refinement_y_max:.1f}] μm (height: {box_height:.1f} μm)")
    print(f"  Distance between nodes: {receiver_center - sender_center:.1f} μm")
    
    return mesh, sender_center, receiver_center, y_center

# =============================================================================
# 2D ADAPTIVE MESH SETUP
# =============================================================================

print("Creating adaptive mesh...")

# Create the adaptive mesh
# You can adjust these parameters for mesh refinement:
# - fine_dx: mesh spacing in refined region (smaller = finer, but slower)
# - coarse_dx: mesh spacing in bulk region (larger = coarser, but faster)
# - box_padding: extra padding around nodes for refined region
# - transition_width: how gradually the mesh transitions from fine to coarse

mesh, sender_center_x, receiver_center_x, sender_center_y = create_adaptive_mesh_for_simulation(
    node_size=node_size,
    sender_center=None,  # Will be calculated automatically
    receiver_center=None,  # Will be calculated automatically
    fine_dx=5.0,  # Fine mesh spacing near nodes
    coarse_dx=40.0,  # Coarse mesh spacing far from nodes
    box_padding=200.0,  # Padding around nodes
    transition_width=100.0,  # Transition width
    total_width=total_width,
    total_height=total_height
)

receiver_center_y = sender_center_y  # Both at same Y position

# Get cell centers
x, y = mesh.cellCenters

print(f"\n2D Simulation Setup:")
print(f"  Mesh: {mesh.numberOfCells} cells (adaptive)")
print(f"  Domain: {total_width} * {total_height} μm²")
print(f"  Node diameter: {node_diameter} μm")
print(f"  Distance: {distance_between} μm (center-to-center)")
print(f"  Sender at: ({sender_center_x:.0f}, {sender_center_y:.0f})")
print(f"  Receiver at: ({receiver_center_x:.0f}, {receiver_center_y:.0f})")
print()


# =============================================================================
# CREATE SMOOTH CONCENTRATION PROFILES
# =============================================================================

# Use adaptive transition width based on finest mesh spacing
# The adaptive mesh has fine_dx = 5.0 μm near nodes
transition_width = 3.0 * 5.0  # 15 μm (3× finest mesh spacing)

print(f"Using smooth tanh profiles with transition_width = {transition_width:.1f} μm")


# =============================================================================
# CELL VARIABLES WITH SMOOTH INITIAL CONDITIONS
# =============================================================================

# Create smooth initial concentration profiles
I2_initial = smooth_circular_profile(x, y, receiver_center_x, receiver_center_y,
                                     node_radius, I2_init, 0.0, transition_width)

'''Calculate total amount'''
total_smooth = calculate_total_amount(I2_initial, mesh)


receiver_mask_boolean = (np.sqrt((x - receiver_center_x)**2 + 
                                 (y - receiver_center_y)**2) <= node_radius)
I2_intial_boolean =np.where(receiver_mask_boolean, I2_init, 0.0)
total_boolean = calculate_total_amount(I2_intial_boolean, mesh)

correction_factor = total_boolean / total_smooth
print(f"Smooth profile has {total_smooth/total_boolean*100:.1f}% of boolean total")
print(f"Correction factor needed: {correction_factor:.3f}")

'''Calculate rest of the values'''

Th2_initial = smooth_circular_profile(x, y, receiver_center_x, receiver_center_y,
                                      node_radius, Th2_init, 0.0, transition_width)

I1O2_initial = smooth_circular_profile(x, y, sender_center_x, sender_center_y,
                                       node_radius, I1O2_init, 0.0, transition_width)

# Create smooth diffusion coefficient profile
# Inside nodes: D_gel (60), Outside nodes: D_solution (150)
D_sender = smooth_circular_profile(x, y, sender_center_x, sender_center_y,
                                   node_radius, D_gel, D_solution, transition_width)

D_receiver = smooth_circular_profile(x, y, receiver_center_x, receiver_center_y,
                                     node_radius, D_gel, D_solution, transition_width)

# Where either node exists, use gel diffusion (take minimum)
D_combined = np.minimum(D_sender, D_receiver)

# Now create CellVariables with smooth initial values
S2 = CellVariable(name="S2", mesh=mesh, value=0.0, hasOld=True)

I2 = CellVariable(name="I2", mesh=mesh, value=I2_initial, hasOld=True)

Th2 = CellVariable(name="Th2", mesh=mesh, value=Th2_initial, hasOld=True)

S2_I2 = CellVariable(name="S2_I2", mesh=mesh, value=0.0, hasOld=True)
S2_Th2 = CellVariable(name="S2_Th2", mesh=mesh, value=0.0, hasOld=True)

I1O2 = CellVariable(name="I1O2", mesh=mesh, value=I1O2_initial)

# Smooth spatially varying diffusion coefficient
D_S2 = CellVariable(name="D_S2", mesh=mesh, value=D_combined)

# =============================================================================
# EQUATIONS (same as before)
# =============================================================================

eq_S2 = (TransientTerm(var=S2) == 
         DiffusionTerm(coeff=D_S2, var=S2) +  
         k_p * I1O2 +
         ImplicitSourceTerm(coeff=-(k_slow * I2 + k_fast * Th2 + k_d_ss), var=S2))

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

eq = eq_S2 & eq_I2 & eq_Th2 & eq_S2_I2 & eq_S2_Th2

# =============================================================================
# FIND RECEIVER CENTER INDEX (for monitoring)
# =============================================================================

# Find cell closest to receiver center
distances_to_receiver = numerix.sqrt((x - receiver_center_x)**2 + 
                                     (y - receiver_center_y)**2)
receiver_center_idx = numerix.argmin(distances_to_receiver)

# Storage
time_points = []
I2_concentration = []
S2_free_concentration = []
S2_total_concentration = []

# =============================================================================
# TIME STEPPING
# =============================================================================

print("Starting 2D simulation with adaptive mesh...")
print(f"Total steps: {n_steps}")

for step in range(n_steps):
    S2.updateOld()
    I2.updateOld()
    Th2.updateOld()
    S2_I2.updateOld()
    S2_Th2.updateOld()
    
    res = 1e10
    sweep = 0
    max_sweeps = 10
    
    while res > 1e-6 and sweep < max_sweeps:
        res = eq.sweep(dt=dt)
        sweep += 1
    
    if step % save_interval_steps == 0:
        current_time = step * dt
        time_points.append(current_time / 3600)
        
        I2_val = I2.value[receiver_center_idx]
        S2_free_val = S2.value[receiver_center_idx]
        S2_total_val = (S2.value[receiver_center_idx] + 
                       S2_I2.value[receiver_center_idx] + 
                       S2_Th2.value[receiver_center_idx])
        
        I2_concentration.append(I2_val)
        S2_free_concentration.append(S2_free_val)
        S2_total_concentration.append(S2_total_val)
        
        if step % (save_interval_steps * 10) == 0:  # Print less frequently
            print(f"t = {current_time/3600:.2f} hr: "
                  f"I2 = {I2_val*1000:.2f} nM, "
                  f"S2_total = {S2_total_val*1000:.2f} nM")

print("\nSimulation complete!")

# =============================================================================
# SAVE RESULTS TO CSV
# =============================================================================

print("Saving results to CSV files...")

# Convert to nM for saving
I2_concentration_nM = np.array(I2_concentration) * 1000
S2_free_concentration_nM = np.array(S2_free_concentration) * 1000
S2_total_concentration_nM = np.array(S2_total_concentration) * 1000

# Create DataFrame
df = pd.DataFrame({
    'Time (hours)': time_points,
    'I2 (nM)': I2_concentration_nM,
    'S2_free (nM)': S2_free_concentration_nM,
    'S2_total (nM)': S2_total_concentration_nM
})

# Save to CSV
csv_filename = f'adaptive_mesh_timeseries_ccd={distance_between:.0f}_dt={dt:.0f}.csv'
df.to_csv(csv_filename, index=False)
print(f"Time series data saved to '{csv_filename}'")

# =============================================================================
# PLOTTING
# =============================================================================

# Time series plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Adaptive Mesh: Domain {total_width/1e3:.0f}mm x {total_height/1e3:.0f}mm, Distance={distance_between:.0f}μm', 
             fontsize=16, fontweight='bold')

axes[0].plot(time_points, I2_concentration_nM, 'b-', linewidth=2)
axes[0].axhline(y=75, color='g', linestyle='--', alpha=0.5, label='75% ON')
axes[0].axhline(y=25, color='r', linestyle='--', alpha=0.5, label='25% OFF')
axes[0].set_xlabel('Time (hours)', fontsize=12)
axes[0].set_ylabel('[I2] (nM)', fontsize=12)
axes[0].set_title(f'I2 at Receiver', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(bottom=0)

axes[1].plot(time_points, S2_free_concentration_nM, 'g-', linewidth=2)
axes[1].set_xlabel('Time (hours)', fontsize=12)
axes[1].set_ylabel('[S2] free (nM)', fontsize=12)
axes[1].set_title('Free S2 at Receiver', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(bottom=0)

axes[2].plot(time_points, S2_total_concentration_nM, 'r-', linewidth=2)
axes[2].set_xlabel('Time (hours)', fontsize=12)
axes[2].set_ylabel('[S2] total (nM)', fontsize=12)
axes[2].set_title('Total S2 at Receiver', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(f'adaptive_mesh_timeseries_ccd={distance_between:.0f}_dt={dt:.0f}.png', 
            dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 2D SPATIAL HEATMAP
# =============================================================================

# print("\nGenerating spatial heatmap...")

# fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# fig.suptitle(f'Adaptive Mesh Spatial Distribution at t={time_points[-1]:.1f} hr', 
#              fontsize=16, fontweight='bold')

# # Get S2 values in nM
# S2_values_nM = S2.value * 1000

# # Plot 1: Scatter plot of S2 concentration (works better with non-uniform mesh)
# x_coords = mesh.cellCenters[0].value
# y_coords = mesh.cellCenters[1].value

# scatter = axes[0].scatter(x_coords, y_coords, c=S2_values_nM, 
#                           cmap='viridis', s=1, vmin=0)
# axes[0].set_xlabel('X position (μm)', fontsize=12)
# axes[0].set_ylabel('Y position (μm)', fontsize=12)
# axes[0].set_title(f'S2 Concentration (nM)', fontsize=14, fontweight='bold')
# axes[0].set_aspect('equal')

# # Mark nodes with circles
# circle_sender = plt.Circle((sender_center_x, sender_center_y), node_radius, 
#                            fill=False, edgecolor='red', linewidth=2, label='Sender')
# circle_receiver = plt.Circle((receiver_center_x, receiver_center_y), node_radius, 
#                              fill=False, edgecolor='blue', linewidth=2, label='Receiver')
# axes[0].add_patch(circle_sender)
# axes[0].add_patch(circle_receiver)
# axes[0].legend(fontsize=10)

# cbar1 = plt.colorbar(scatter, ax=axes[0])
# cbar1.set_label('[S2] (nM)', fontsize=11)

# # Plot 2: Cross-section along line between nodes
# # Find cells closest to the horizontal line at y = sender_center_y
# y_tolerance = 50  # μm tolerance for selecting cells near the line
# line_mask = np.abs(y_coords - sender_center_y) < y_tolerance
# x_line = x_coords[line_mask]
# S2_line = S2_values_nM[line_mask]

# # Sort by x coordinate
# sort_idx = np.argsort(x_line)
# x_line_sorted = x_line[sort_idx]
# S2_line_sorted = S2_line[sort_idx]

# axes[1].plot(x_line_sorted, S2_line_sorted, 'b-', linewidth=1, alpha=0.7)
# axes[1].axvline(x=sender_center_x, color='red', linestyle='--', linewidth=2, 
#                 alpha=0.7, label='Sender')
# axes[1].axvline(x=receiver_center_x, color='blue', linestyle='--', linewidth=2,
#                 alpha=0.7, label='Receiver')
# axes[1].set_xlabel('X position (μm)', fontsize=12)
# axes[1].set_ylabel('[S2] (nM)', fontsize=12)
# axes[1].set_title('S2 Profile Along Line Between Nodes', fontsize=14, fontweight='bold')
# axes[1].grid(True, alpha=0.3)
# axes[1].legend(fontsize=10)
# axes[1].set_ylim(bottom=0)
# axes[1].set_xlim([0, total_width])

# plt.tight_layout()
# plt.savefig(f'adaptive_mesh_spatial_ccd={distance_between:.0f}.png', 
#             dpi=300, bbox_inches='tight')
# plt.show()

# print("\nAll plots saved successfully!")