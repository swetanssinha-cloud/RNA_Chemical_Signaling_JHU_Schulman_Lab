import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, ImplicitSourceTerm
from fipy.tools import numerix
import csv
import pandas as pd

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
distance_between = 1500 # Test at large distance
total_width = 1e4 #10000 μm = 1 cm
total_height = 1e3 #1000 μm = 1 mm

dt = 30.0
total_time = 8 * 3600
n_steps = int(total_time / dt)
save_interval_time = 60.0
save_interval_steps = int(save_interval_time / dt)

# =============================================================================
# ADAPTIVE MESH GENERATION FUNCTIONS
# =============================================================================

def create_adaptive_mesh_for_simulation(
    node_size=50.0,
    sender_center=None,
    receiver_center=None,
    fine_dx=5.0,
    coarse_dx=40.0,
    node_radius=37.5,
    transition_distance=300.0,  # Distance over which to transition from fine to coarse
    total_width=1e4,
    total_height=1e3
):
    """
    Create adaptive mesh for the 2D genelet simulation with radial refinement around nodes.
    
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
    node_radius : float
        Radius of the nodes (μm)
    transition_distance : float
        Distance from node surface over which to transition from fine to coarse mesh (μm)
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
    
    # Node centers
    sender_pos = np.array([sender_center, total_height / 2])
    receiver_pos = np.array([receiver_center, total_height / 2])
    
    def distance_to_nearest_node(x, y):
        """
        Calculate the minimum distance from point (x,y) to the nearest node surface.
        Returns the distance from the node surface (not center).
        """
        # Distance to sender center
        dist_to_sender = np.sqrt((x - sender_pos[0])**2 + (y - sender_pos[1])**2)
        # Distance to receiver center
        dist_to_receiver = np.sqrt((x - receiver_pos[0])**2 + (y - receiver_pos[1])**2)
        
        # Distance from node surface (subtract node radius)
        dist_from_sender_surface = np.maximum(dist_to_sender - node_radius, 0)
        dist_from_receiver_surface = np.maximum(dist_to_receiver - node_radius, 0)
        
        # Return minimum distance to either node surface
        return np.minimum(dist_from_sender_surface, dist_from_receiver_surface)

    def calculate_refinement_factor(x, y):
        """
        Calculate refinement factor based on distance to nearest node.
        Returns value between 0 (fine, at node) and 1 (coarse, far away).
        Uses smooth tanh transition.
        """
        dist = distance_to_nearest_node(x, y)
        
        # Smooth transition using tanh
        # At dist=0 (node surface): blend ≈ 0 (fine mesh)
        # At dist=transition_distance: blend ≈ 1 (coarse mesh)
        blend = 0.5 * (1 + np.tanh((dist - transition_distance/2) / (transition_distance/6)))
        
        return blend
    
    def create_adaptive_spacing_1D(total_length, positions_other_dim,
                                    fine_dx, coarse_dx, is_x_direction=True):
        """
        Create 1D spacing array that depends on position in both dimensions.
        Now uses radial distance to nodes instead of rectangular boxes.
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
    
    print(f"\nRadial Adaptive Mesh Created:")
    print(f"  Total cells: {mesh.numberOfCells}")
    print(f"  X cells: {len(dx_array)}")
    print(f"  Y cells: {len(dy_array)}")
    print(f"  Min dx: {dx_array.min():.2f} μm")
    print(f"  Max dx: {dx_array.max():.2f} μm")
    print(f"  Min dy: {dy_array.min():.2f} μm")
    print(f"  Max dy: {dy_array.max():.2f} μm")
    print(f"\nRefinement parameters:")
    print(f"  Node radius: {node_radius:.1f} μm")
    print(f"  Transition distance: {transition_distance:.1f} μm")
    print(f"  Fine mesh (at nodes): {fine_dx:.1f} μm")
    print(f"  Coarse mesh (far field): {coarse_dx:.1f} μm")
    print(f"  Distance between nodes: {receiver_center - sender_center:.1f} μm")
    
    return mesh, sender_center, receiver_center, total_height / 2
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
    fine_dx=5.0,  # Fine mesh spacing at node surface
    coarse_dx=40.0,  # Coarse mesh spacing far from nodes
    node_radius=node_radius,  # Use the actual node radius
    transition_distance=300.0,  # Distance over which mesh transitions from fine to coarse
    total_width=total_width,
    total_height=total_height
)

receiver_center_y = sender_center_y  # Both at same Y position

# =============================================================================
# MESH AND SYSTEM VISUALIZATION
# =============================================================================

def visualize_mesh_and_system(mesh, sender_center_x, receiver_center_x, 
                               sender_center_y, receiver_center_y, 
                               node_size, node_radius):
    """
    Create comprehensive visualization of the adaptive mesh and system geometry.
    """
    x_coords = mesh.cellCenters[0].value
    y_coords = mesh.cellCenters[1].value
    
    # Create sender and receiver masks
    sender_mask = (np.sqrt((x_coords - sender_center_x)**2 + 
                           (y_coords - sender_center_y)**2) <= node_radius)
    receiver_mask = (np.sqrt((x_coords - receiver_center_x)**2 + 
                             (y_coords - receiver_center_y)**2) <= node_radius)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # =========================================================================
    # 1. Full mesh structure with nodes
    # =========================================================================
    ax = axes[0, 0]
    
    # Plot all cell centers
    ax.scatter(x_coords, y_coords, c='lightgray', s=1, alpha=0.5, label='Mesh cells')
    
    # Highlight sender and receiver
    ax.scatter(x_coords[sender_mask], y_coords[sender_mask], 
               c='blue', s=5, label='Sender node', alpha=0.7)
    ax.scatter(x_coords[receiver_mask], y_coords[receiver_mask], 
               c='red', s=5, label='Receiver node', alpha=0.7)
    
    # Draw circles around nodes
    sender_circle = plt.Circle((sender_center_x, sender_center_y), node_radius,
                               fill=False, color='blue', linewidth=2, linestyle='--')
    receiver_circle = plt.Circle((receiver_center_x, receiver_center_y), node_radius,
                                 fill=False, color='red', linewidth=2, linestyle='--')
    ax.add_patch(sender_circle)
    ax.add_patch(receiver_circle)
    
    # Add distance annotation
    ax.plot([sender_center_x, receiver_center_x], 
            [sender_center_y, receiver_center_y],
            'k--', linewidth=1, alpha=0.5)
    mid_x = (sender_center_x + receiver_center_x) / 2
    mid_y = (sender_center_y + receiver_center_y) / 2
    distance = receiver_center_x - sender_center_x
    ax.text(mid_x, mid_y + 100, f'{distance:.0f} μm', 
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('X Position (μm)', fontsize=12)
    ax.set_ylabel('Y Position (μm)', fontsize=12)
    ax.set_title('Full Mesh Structure with Nodes', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # 2. Zoomed view of sender node
    # =========================================================================
    ax = axes[0, 1]
    
    zoom_range = node_radius * 3
    x_min_zoom = sender_center_x - zoom_range
    x_max_zoom = sender_center_x + zoom_range
    y_min_zoom = sender_center_y - zoom_range
    y_max_zoom = sender_center_y + zoom_range
    
    # Filter points in zoom range
    zoom_mask = ((x_coords >= x_min_zoom) & (x_coords <= x_max_zoom) &
                 (y_coords >= y_min_zoom) & (y_coords <= y_max_zoom))
    
    ax.scatter(x_coords[zoom_mask], y_coords[zoom_mask], 
               c='lightgray', s=20, alpha=0.5)
    ax.scatter(x_coords[sender_mask], y_coords[sender_mask], 
               c='blue', s=30, label='Sender node', alpha=0.8)
    
    sender_circle_zoom = plt.Circle((sender_center_x, sender_center_y), node_radius,
                                    fill=False, color='blue', linewidth=2)
    ax.add_patch(sender_circle_zoom)
    
    ax.set_xlabel('X Position (μm)', fontsize=12)
    ax.set_ylabel('Y Position (μm)', fontsize=12)
    ax.set_title(f'Zoomed View: Sender Node (radius={node_radius:.1f} μm)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([x_min_zoom, x_max_zoom])
    ax.set_ylim([y_min_zoom, y_max_zoom])
    
    # =========================================================================
    # 3. Zoomed view of receiver node
    # =========================================================================
    ax = axes[1, 0]
    
    x_min_zoom = receiver_center_x - zoom_range
    x_max_zoom = receiver_center_x + zoom_range
    y_min_zoom = receiver_center_y - zoom_range
    y_max_zoom = receiver_center_y + zoom_range
    
    zoom_mask = ((x_coords >= x_min_zoom) & (x_coords <= x_max_zoom) &
                 (y_coords >= y_min_zoom) & (y_coords <= y_max_zoom))
    
    ax.scatter(x_coords[zoom_mask], y_coords[zoom_mask], 
               c='lightgray', s=20, alpha=0.5)
    ax.scatter(x_coords[receiver_mask], y_coords[receiver_mask], 
               c='red', s=30, label='Receiver node', alpha=0.8)
    
    receiver_circle_zoom = plt.Circle((receiver_center_x, receiver_center_y), node_radius,
                                      fill=False, color='red', linewidth=2)
    ax.add_patch(receiver_circle_zoom)
    
    ax.set_xlabel('X Position (μm)', fontsize=12)
    ax.set_ylabel('Y Position (μm)', fontsize=12)
    ax.set_title(f'Zoomed View: Receiver Node (radius={node_radius:.1f} μm)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([x_min_zoom, x_max_zoom])
    ax.set_ylim([y_min_zoom, y_max_zoom])
    
    # =========================================================================
    # 4. Mesh resolution heatmap
    # =========================================================================
    ax = axes[1, 1]
    
    # Calculate local mesh resolution (average of dx and dy near each cell)
    # This is approximate - showing cell density
    from scipy.spatial import cKDTree
    
    points = np.column_stack([x_coords, y_coords])
    tree = cKDTree(points)
    
    # Find 5 nearest neighbors for each point
    distances, _ = tree.query(points, k=6)  # k=6 because first is itself
    avg_distance = distances[:, 1:].mean(axis=1)  # Average of 5 nearest
    
    scatter = ax.scatter(x_coords, y_coords, c=avg_distance, 
                         s=3, cmap='RdYlGn_r', alpha=0.8)
    
    # Highlight nodes
    ax.scatter(x_coords[sender_mask], y_coords[sender_mask], 
               c='blue', s=10, edgecolors='white', linewidths=0.5, 
               label='Sender', zorder=5)
    ax.scatter(x_coords[receiver_mask], y_coords[receiver_mask], 
               c='red', s=10, edgecolors='white', linewidths=0.5,
               label='Receiver', zorder=5)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Avg. Cell Spacing (μm)', fontsize=11)
    
    ax.set_xlabel('X Position (μm)', fontsize=12)
    ax.set_ylabel('Y Position (μm)', fontsize=12)
    ax.set_title('Mesh Resolution Heatmap', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adaptive_mesh_visualization.png', dpi=300, bbox_inches='tight')
    print("\nMesh visualization saved as 'adaptive_mesh_visualization.png'")
    plt.show()
    
    return fig

# =============================================================================
# Additional: 1D cross-section plots
# =============================================================================
# =============================================================================
# IMPROVED MESH VISUALIZATION - Shows actual refinement strategy
# =============================================================================

def visualize_mesh_and_system_improved(mesh, sender_center_x, receiver_center_x, 
                                       sender_center_y, receiver_center_y, 
                                       node_size, node_radius, transition_distance):
    """
    Create comprehensive visualization with TRUE refinement factor shown.
    """
    x_coords = mesh.cellCenters[0].value
    y_coords = mesh.cellCenters[1].value
    
    # Recreate the refinement calculation for visualization
    sender_pos = np.array([sender_center_x, sender_center_y])
    receiver_pos = np.array([receiver_center_x, receiver_center_y])
    
    def distance_to_nearest_node_surface(x, y):
        dist_to_sender = np.sqrt((x - sender_pos[0])**2 + (y - sender_pos[1])**2)
        dist_to_receiver = np.sqrt((x - receiver_pos[0])**2 + (y - receiver_pos[1])**2)
        dist_from_sender_surface = np.maximum(dist_to_sender - node_radius, 0)
        dist_from_receiver_surface = np.maximum(dist_to_receiver - node_radius, 0)
        return np.minimum(dist_from_sender_surface, dist_from_receiver_surface)
    
    def calculate_refinement_factor(x, y):
        dist = distance_to_nearest_node_surface(x, y)
        blend = 0.5 * (1 + np.tanh((dist - transition_distance/2) / (transition_distance/6)))
        return blend
    
    # Calculate refinement factor for each cell
    refinement_factors = np.array([calculate_refinement_factor(x, y) 
                                   for x, y in zip(x_coords, y_coords)])
    
    # Calculate actual intended mesh size at each location
    fine_dx = 5.0
    coarse_dx = 40.0
    intended_mesh_size = fine_dx + (coarse_dx - fine_dx) * refinement_factors
    
    # Create sender and receiver masks
    sender_mask = (np.sqrt((x_coords - sender_center_x)**2 + 
                           (y_coords - sender_center_y)**2) <= node_radius)
    receiver_mask = (np.sqrt((x_coords - receiver_center_x)**2 + 
                             (y_coords - receiver_center_y)**2) <= node_radius)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # =========================================================================
    # 1. Full mesh structure with nodes
    # =========================================================================
    ax = axes[0, 0]
    
    # Subsample for visibility
    subsample = max(1, len(x_coords) // 50000)
    
    ax.scatter(x_coords[::subsample], y_coords[::subsample], 
               c='lightgray', s=1, alpha=0.5, label='Mesh cells')
    
    # Highlight sender and receiver
    ax.scatter(x_coords[sender_mask], y_coords[sender_mask], 
               c='blue', s=5, label='Sender node', alpha=0.7)
    ax.scatter(x_coords[receiver_mask], y_coords[receiver_mask], 
               c='red', s=5, label='Receiver node', alpha=0.7)
    
    # Draw circles around nodes and transition zones
    sender_circle = plt.Circle((sender_center_x, sender_center_y), node_radius,
                               fill=False, color='blue', linewidth=2, linestyle='-', label='Node boundary')
    receiver_circle = plt.Circle((receiver_center_x, receiver_center_y), node_radius,
                                 fill=False, color='red', linewidth=2, linestyle='-')
    
    sender_transition = plt.Circle((sender_center_x, sender_center_y), 
                                   node_radius + transition_distance,
                                   fill=False, color='blue', linewidth=1.5, 
                                   linestyle='--', alpha=0.5, label='Transition zone')
    receiver_transition = plt.Circle((receiver_center_x, receiver_center_y), 
                                     node_radius + transition_distance,
                                     fill=False, color='red', linewidth=1.5, 
                                     linestyle='--', alpha=0.5)
    
    ax.add_patch(sender_circle)
    ax.add_patch(receiver_circle)
    ax.add_patch(sender_transition)
    ax.add_patch(receiver_transition)
    
    # Add distance annotation
    ax.plot([sender_center_x, receiver_center_x], 
            [sender_center_y, receiver_center_y],
            'k--', linewidth=1, alpha=0.5)
    mid_x = (sender_center_x + receiver_center_x) / 2
    mid_y = (sender_center_y + receiver_center_y) / 2
    distance = receiver_center_x - sender_center_x
    ax.text(mid_x, mid_y + 100, f'{distance:.0f} μm', 
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('X Position (μm)', fontsize=12)
    ax.set_ylabel('Y Position (μm)', fontsize=12)
    ax.set_title('Full Mesh Structure with Radial Refinement Zones', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # 2. Zoomed view of sender node
    # =========================================================================
    ax = axes[0, 1]
    
    zoom_range = node_radius * 4
    x_min_zoom = sender_center_x - zoom_range
    x_max_zoom = sender_center_x + zoom_range
    y_min_zoom = sender_center_y - zoom_range
    y_max_zoom = sender_center_y + zoom_range
    
    zoom_mask = ((x_coords >= x_min_zoom) & (x_coords <= x_max_zoom) &
                 (y_coords >= y_min_zoom) & (y_coords <= y_max_zoom))
    
    ax.scatter(x_coords[zoom_mask], y_coords[zoom_mask], 
               c='lightgray', s=20, alpha=0.5)
    ax.scatter(x_coords[sender_mask], y_coords[sender_mask], 
               c='blue', s=30, label='Sender node', alpha=0.8)
    
    sender_circle_zoom = plt.Circle((sender_center_x, sender_center_y), node_radius,
                                    fill=False, color='blue', linewidth=2)
    sender_transition_zoom = plt.Circle((sender_center_x, sender_center_y), 
                                        node_radius + transition_distance,
                                        fill=False, color='blue', linewidth=1.5, 
                                        linestyle='--', alpha=0.5)
    ax.add_patch(sender_circle_zoom)
    ax.add_patch(sender_transition_zoom)
    
    ax.set_xlabel('X Position (μm)', fontsize=12)
    ax.set_ylabel('Y Position (μm)', fontsize=12)
    ax.set_title(f'Sender: Node + Transition Zone (r={node_radius:.1f}+{transition_distance:.1f} μm)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([x_min_zoom, x_max_zoom])
    ax.set_ylim([y_min_zoom, y_max_zoom])
    
    # =========================================================================
    # 3. Intended mesh size heatmap (THE TRUE REFINEMENT)
    # =========================================================================
    ax = axes[1, 0]
    
    scatter = ax.scatter(x_coords, y_coords, c=intended_mesh_size, 
                         s=3, cmap='RdYlGn_r', alpha=0.8, vmin=fine_dx, vmax=coarse_dx)
    
    # Highlight nodes
    ax.scatter(x_coords[sender_mask], y_coords[sender_mask], 
               c='blue', s=10, edgecolors='white', linewidths=0.5, 
               label='Sender', zorder=5)
    ax.scatter(x_coords[receiver_mask], y_coords[receiver_mask], 
               c='red', s=10, edgecolors='white', linewidths=0.5,
               label='Receiver', zorder=5)
    
    # Add transition circles
    sender_transition = plt.Circle((sender_center_x, sender_center_y), 
                                   node_radius + transition_distance,
                                   fill=False, color='white', linewidth=2, 
                                   linestyle='--', alpha=0.8)
    receiver_transition = plt.Circle((receiver_center_x, receiver_center_y), 
                                     node_radius + transition_distance,
                                     fill=False, color='white', linewidth=2, 
                                     linestyle='--', alpha=0.8)
    ax.add_patch(sender_transition)
    ax.add_patch(receiver_transition)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Target Mesh Size (μm)', fontsize=11)
    
    ax.set_xlabel('X Position (μm)', fontsize=12)
    ax.set_ylabel('Y Position (μm)', fontsize=12)
    ax.set_title('Intended Mesh Refinement (Radial from Nodes)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # 4. Distance to nearest node surface
    # =========================================================================
    ax = axes[1, 1]
    
    distances = np.array([distance_to_nearest_node_surface(x, y) 
                         for x, y in zip(x_coords, y_coords)])
    
    scatter = ax.scatter(x_coords, y_coords, c=distances, 
                         s=3, cmap='viridis', alpha=0.8)
    
    # Highlight nodes
    ax.scatter(x_coords[sender_mask], y_coords[sender_mask], 
               c='blue', s=10, edgecolors='white', linewidths=0.5, 
               label='Sender', zorder=5)
    ax.scatter(x_coords[receiver_mask], y_coords[receiver_mask], 
               c='red', s=10, edgecolors='white', linewidths=0.5,
               label='Receiver', zorder=5)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Distance to Nearest Node Surface (μm)', fontsize=11)
    
    ax.set_xlabel('X Position (μm)', fontsize=12)
    ax.set_ylabel('Y Position (μm)', fontsize=12)
    ax.set_title('Distance Field (Radial from Node Surfaces)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adaptive_mesh_visualization_radial.png', dpi=300, bbox_inches='tight')
    print("\nImproved mesh visualization saved as 'adaptive_mesh_visualization_radial.png'")
    plt.show()
    
    return fig

# Call the improved visualization
print("\nGenerating IMPROVED mesh visualizations...")

visualize_mesh_and_system_improved(mesh, sender_center_x, receiver_center_x,
                                   sender_center_y, receiver_center_y,
                                   node_size, node_radius, transition_distance=300.0)

# Keep the 1D plots (they're already correct)
# plot_mesh_spacing_1D(mesh, sender_center_x, receiver_center_x,
#                      sender_center_y, total_width, total_height)

print("\nVisualization complete!")