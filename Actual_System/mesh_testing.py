# # =============================================================================
# # NON-UNIFORM MESH - 2D
# # =============================================================================

# from fipy import Grid2D
# import numpy as np
# import matplotlib.pyplot as plt

# def create_nonuniform_mesh_2D(L_total, x_fine_left, x_fine_right,
#                               y_fine_bottom, y_fine_top,
#                               dx_fine=10.0, dx_coarse=100.0, ):
#     """
#     Create 2D mesh with fine region in center, coarse at edges.
#     """
    
#     # X-direction spacing
#     n_coarse_left_x = int(x_fine_left / dx_coarse)
#     n_fine_x = int((x_fine_right - x_fine_left) / dx_fine)
#     n_coarse_right_x = int((L_total - x_fine_right) / dx_coarse)
    
#     x_left = np.linspace(0, x_fine_left, n_coarse_left_x, endpoint=False)
#     x_fine = np.linspace(x_fine_left, x_fine_right, n_fine_x, endpoint=False)
#     x_right = np.linspace(x_fine_right, L_total, n_coarse_right_x + 1)
#     x_all = np.concatenate([x_left, x_fine, x_right])
#     dx_array = np.diff(x_all)
    
#     # Y-direction spacing (symmetric)
#     n_coarse_bottom_y = int(y_fine_bottom / dx_coarse)
#     n_fine_y = int((y_fine_top - y_fine_bottom) / dx_fine)
#     n_coarse_top_y = int((L_total - y_fine_top) / dx_coarse)
    
#     y_bottom = np.linspace(0, y_fine_bottom, n_coarse_bottom_y, endpoint=False)
#     y_fine = np.linspace(y_fine_bottom, y_fine_top, n_fine_y, endpoint=False)
#     y_top = np.linspace(y_fine_top, L_total, n_coarse_top_y + 1)
#     y_all = np.concatenate([y_bottom, y_fine, y_top])
#     dy_array = np.diff(y_all)
    
#     print(f"\n2D Non-uniform mesh created:")
#     print(f"  X-direction: {len(dx_array)} cells")
#     print(f"    Left coarse:  {n_coarse_left_x} × {dx_coarse:.1f} μm")
#     print(f"    Fine:         {n_fine_x} × {dx_fine:.1f} μm")
#     print(f"    Right coarse: {n_coarse_right_x} × {dx_coarse:.1f} μm")
#     print(f"  Y-direction: {len(dy_array)} cells")
#     print(f"    Bottom coarse: {n_coarse_bottom_y} × {dx_coarse:.1f} μm")
#     print(f"    Fine:          {n_fine_y} × {dx_fine:.1f} μm")
#     print(f"    Top coarse:    {n_coarse_top_y} × {dx_coarse:.1f} μm")
#     print(f"  Total cells: {len(dx_array) * len(dy_array)}")
#     print(f"  Domain: {L_total:.1f} × {L_total:.1f} μm²")
    
#     return Grid2D(dx=dx_array, dy=dy_array)

# # Set up 2D domain
# total_length = 5000.0  # Larger domain
# center_x = total_length / 2
# center_y = total_length / 2
# distance_between = 1000
# node_size = 50

# sender_center_x = center_x - distance_between / 2
# sender_center_y = center_y
# receiver_center_x = center_x + distance_between / 2
# receiver_center_y = center_y

# # Fine region around both nodes
# buffer = 300.0
# x_fine_left = sender_center_x - buffer
# x_fine_right = receiver_center_x + buffer
# y_fine_bottom = center_y - buffer
# y_fine_top = center_y + buffer

# mesh = create_nonuniform_mesh_2D(
#     L_total=total_length,
#     x_fine_left=x_fine_left,
#     x_fine_right=x_fine_right,
#     y_fine_bottom=y_fine_bottom,
#     y_fine_top=y_fine_top,
#     dx_fine=10.0,    # 10 μm near nodes
#     dx_coarse=100.0  # 100 μm at edges
# )

# x, y = mesh.cellCenters

# # Define node masks (same as before)
# sender_mask = ((x >= sender_center_x - node_size/2) & 
#                (x <= sender_center_x + node_size/2) &
#                (y >= sender_center_y - node_size/2) & 
#                (y <= sender_center_y + node_size/2))

# receiver_mask = ((x >= receiver_center_x - node_size/2) & 
#                  (x <= receiver_center_x + node_size/2) &
#                  (y >= receiver_center_y - node_size/2) & 
#                  (y <= receiver_center_y + node_size/2))

# gel_mask = sender_mask | receiver_mask


# =============================================================================
# VISUALIZE THE MESH
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from fipy import Grid2D

def create_and_visualize_2D_nonuniform_mesh(
    node_size=50.0,
    sender_center=1000.0,
    receiver_center=2500.0,
    fine_dx=5.0,
    coarse_dx=50.0,
    transition_width=200.0,
    total_length=5000.0
):
    """
    Create and visualize a 2D non-uniform mesh for the tethered genelet system.
    
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
    transition_width : float
        Width of transition region between fine and coarse (μm)
    total_length : float
        Total domain size (μm)
    """
    
    def create_graded_spacing_1D(total_length, node_centers, node_size, 
                                  fine_dx, coarse_dx, transition_width):
        """Create 1D array of cell spacings with refinement near nodes."""
        positions = [0.0]
        current_pos = 0.0
        
        while current_pos < total_length:
            # Find minimum distance to any node
            min_dist_to_node = min(
                abs(current_pos - center) for center in node_centers
            )
            
            # Distance from nearest node edge
            dist_from_edge = max(0, min_dist_to_node - node_size/2)
            
            # Smooth transition using tanh
            if dist_from_edge < transition_width:
                blend = 0.5 * (1 + np.tanh(
                    (dist_from_edge - transition_width/2) / (transition_width/10)
                ))
                dx = fine_dx + (coarse_dx - fine_dx) * blend
            else:
                dx = coarse_dx
            
            current_pos += dx
            if current_pos < total_length:
                positions.append(current_pos)
        
        # Convert positions to spacing array
        positions = np.array(positions)
        dx_array = np.diff(positions)
        
        return dx_array
    
    # Create spacing arrays for x and y directions
    node_centers_x = [sender_center, receiver_center]
    node_centers_y = [total_length/2]  # Center vertically
    
    dx_array = create_graded_spacing_1D(
        total_length, node_centers_x, node_size, 
        fine_dx, coarse_dx, transition_width
    )
    
    dy_array = create_graded_spacing_1D(
        total_length, node_centers_y, node_size,
        fine_dx, coarse_dx, transition_width
    )
    
    # Create the mesh
    mesh = Grid2D(dx=dx_array, dy=dy_array)
    
    print(f"Mesh created:")
    print(f"  Total cells: {mesh.numberOfCells}")
    print(f"  X cells: {len(dx_array)}")
    print(f"  Y cells: {len(dy_array)}")
    print(f"  Min dx: {dx_array.min():.2f} μm")
    print(f"  Max dx: {dx_array.max():.2f} μm")
    print(f"  Min dy: {dy_array.min():.2f} μm")
    print(f"  Max dy: {dy_array.max():.2f} μm")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Full mesh structure
    ax = axes[0, 0]
    x_coords = mesh.cellCenters[0].value
    y_coords = mesh.cellCenters[1].value
    ax.plot(x_coords, y_coords, 'k.', markersize=0.5, alpha=0.3)
    
    # Mark node regions
    for center in node_centers_x:
        rect = plt.Rectangle(
            (center - node_size/2, total_length/2 - node_size/2),
            node_size, node_size,
            fill=False, edgecolor='red', linewidth=2
        )
        ax.add_patch(rect)
    
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title('Full Mesh Structure (cell centers)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 2. Zoomed view near sender
    ax = axes[0, 1]
    zoom_range = 300
    mask = ((x_coords > sender_center - zoom_range) & 
            (x_coords < sender_center + zoom_range) &
            (y_coords > total_length/2 - zoom_range) & 
            (y_coords < total_length/2 + zoom_range))
    ax.plot(x_coords[mask], y_coords[mask], 'k.', markersize=2)
    
    rect = plt.Rectangle(
        (sender_center - node_size/2, total_length/2 - node_size/2),
        node_size, node_size,
        fill=False, edgecolor='red', linewidth=2
    )
    ax.add_patch(rect)
    
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title(f'Mesh Near Sender (±{zoom_range} μm)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 3. Cell spacing vs position (X direction)
    ax = axes[1, 0]
    x_cell_positions = np.cumsum(np.concatenate([[0], dx_array]))
    x_cell_centers = 0.5 * (x_cell_positions[:-1] + x_cell_positions[1:])
    
    ax.plot(x_cell_centers, dx_array, 'b-', linewidth=1)
    ax.axhline(y=fine_dx, color='g', linestyle='--', label=f'Fine: {fine_dx} μm')
    ax.axhline(y=coarse_dx, color='r', linestyle='--', label=f'Coarse: {coarse_dx} μm')
    
    for center in node_centers_x:
        ax.axvline(x=center, color='k', linestyle=':', alpha=0.5)
        ax.axvspan(center - node_size/2, center + node_size/2, 
                   alpha=0.2, color='red')
    
    ax.set_xlabel('X Position (μm)')
    ax.set_ylabel('Cell Spacing dx (μm)')
    ax.set_title('Cell Spacing in X Direction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Cell spacing vs position (Y direction)
    ax = axes[1, 1]
    y_cell_positions = np.cumsum(np.concatenate([[0], dy_array]))
    y_cell_centers = 0.5 * (y_cell_positions[:-1] + y_cell_positions[1:])
    
    ax.plot(y_cell_centers, dy_array, 'b-', linewidth=1)
    ax.axhline(y=fine_dx, color='g', linestyle='--', label=f'Fine: {fine_dx} μm')
    ax.axhline(y=coarse_dx, color='r', linestyle='--', label=f'Coarse: {coarse_dx} μm')
    
    for center in node_centers_y:
        ax.axvline(x=center, color='k', linestyle=':', alpha=0.5)
        ax.axvspan(center - node_size/2, center + node_size/2, 
                   alpha=0.2, color='red')
    
    ax.set_xlabel('Y Position (μm)')
    ax.set_ylabel('Cell Spacing dy (μm)')
    ax.set_title('Cell Spacing in Y Direction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mesh_visualization_2D.png', dpi=300, bbox_inches='tight')
    print("\nMesh visualization saved as 'mesh_visualization_2D.png'")
    plt.show()
    
    return mesh, dx_array, dy_array


# Example usage with default parameters
if __name__ == "__main__":
    mesh, dx_array, dy_array = create_and_visualize_2D_nonuniform_mesh(
        node_size=50.0,
        sender_center=1000.0,
        receiver_center=2500.0,  # 1500 μm distance
        fine_dx=5.0,
        coarse_dx=50.0,
        transition_width=200.0,
        total_length=5000.0
    )


#===========================================
#v 2 below (box around nodes)
import numpy as np
import matplotlib.pyplot as plt
from fipy import Grid2D

def create_and_visualize_2D_nonuniform_mesh(
    node_size=50.0,
    sender_center=1000.0,
    receiver_center=2500.0,
    fine_dx=5.0,
    coarse_dx=50.0,
    refinement_region_size=300.0,  # Box size around nodes
    transition_width=100.0,
    total_length=5000.0
):
    """
    Create and visualize a 2D non-uniform mesh with localized refinement.
    
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
    refinement_region_size : float
        Size of square region around each node to refine (μm)
    transition_width : float
        Width of transition between fine and coarse (μm)
    total_length : float
        Total domain size (μm)
    """
    
    def calculate_refinement_factor_2D(x, y, node_centers_x, node_centers_y, 
                                       refinement_region_size, transition_width):
        """
        Calculate refinement factor based on 2D distance to nearest node.
        Returns value between 0 (fine) and 1 (coarse).
        """
        min_dist = float('inf')
        
        # Find minimum distance to any node center
        for nx in node_centers_x:
            for ny in node_centers_y:
                dist = np.sqrt((x - nx)**2 + (y - ny)**2)
                min_dist = min(min_dist, dist)
        
        # Distance from refinement region boundary
        dist_from_boundary = max(0, min_dist - refinement_region_size/2)
        
        # Smooth transition using tanh
        if dist_from_boundary < transition_width:
            blend = 0.5 * (1 + np.tanh(
                (dist_from_boundary - transition_width/2) / (transition_width/10)
            ))
        else:
            blend = 1.0
        
        return blend
    
    def create_adaptive_spacing_1D(total_length, positions_other_dim, 
                                    node_centers_x, node_centers_y,
                                    refinement_region_size, transition_width,
                                    fine_dx, coarse_dx, is_x_direction=True):
        """
        Create 1D spacing array that depends on position in both dimensions.
        """
        positions = [0.0]
        current_pos = 0.0
        
        while current_pos < total_length:
            # Sample refinement at current position
            # Average over the other dimension to get representative spacing
            refinement_samples = []
            for other_pos in positions_other_dim[::max(1, len(positions_other_dim)//20)]:
                if is_x_direction:
                    x, y = current_pos, other_pos
                else:
                    x, y = other_pos, current_pos
                
                blend = calculate_refinement_factor_2D(
                    x, y, node_centers_x, node_centers_y,
                    refinement_region_size, transition_width
                )
                refinement_samples.append(blend)
            
            # Use minimum blend (finest mesh needed along this line)
            blend = min(refinement_samples)
            dx = fine_dx + (coarse_dx - fine_dx) * blend
            
            current_pos += dx
            if current_pos < total_length:
                positions.append(current_pos)
        
        positions = np.array(positions)
        dx_array = np.diff(positions)
        return positions, dx_array
    
    # Node locations
    node_centers_x = [sender_center, receiver_center]
    node_centers_y = [total_length/2]  # Center vertically
    
    # First pass: create preliminary Y spacing with uniform assumption
    y_positions_prelim = np.arange(0, total_length, coarse_dx)
    
    # Create X spacing considering all Y positions
    x_positions, dx_array = create_adaptive_spacing_1D(
        total_length, y_positions_prelim,
        node_centers_x, node_centers_y,
        refinement_region_size, transition_width,
        fine_dx, coarse_dx, is_x_direction=True
    )
    
    # Create Y spacing considering all X positions
    y_positions, dy_array = create_adaptive_spacing_1D(
        total_length, x_positions,
        node_centers_x, node_centers_y,
        refinement_region_size, transition_width,
        fine_dx, coarse_dx, is_x_direction=False
    )
    
    # Create the mesh
    mesh = Grid2D(dx=dx_array, dy=dy_array)
    
    print(f"Mesh created:")
    print(f"  Total cells: {mesh.numberOfCells}")
    print(f"  X cells: {len(dx_array)}")
    print(f"  Y cells: {len(dy_array)}")
    print(f"  Min dx: {dx_array.min():.2f} μm")
    print(f"  Max dx: {dx_array.max():.2f} μm")
    print(f"  Min dy: {dy_array.min():.2f} μm")
    print(f"  Max dy: {dy_array.max():.2f} μm")
    print(f"  Refinement region size: {refinement_region_size} μm around each node")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Full mesh structure
    ax = axes[0, 0]
    x_coords = mesh.cellCenters[0].value
    y_coords = mesh.cellCenters[1].value
    ax.plot(x_coords, y_coords, 'k.', markersize=0.5, alpha=0.3)
    
    # Mark refinement regions and node regions
    for nx in node_centers_x:
        for ny in node_centers_y:
            # Refinement region
            refinement_rect = plt.Rectangle(
                (nx - refinement_region_size/2, ny - refinement_region_size/2),
                refinement_region_size, refinement_region_size,
                fill=False, edgecolor='blue', linewidth=1, linestyle='--',
                label='Refinement region' if nx == node_centers_x[0] else ''
            )
            ax.add_patch(refinement_rect)
            
            # Node region
            node_rect = plt.Rectangle(
                (nx - node_size/2, ny - node_size/2),
                node_size, node_size,
                fill=False, edgecolor='red', linewidth=2,
                label='Node' if nx == node_centers_x[0] else ''
            )
            ax.add_patch(node_rect)
    
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title('Full Mesh Structure (cell centers)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Zoomed view near sender
    ax = axes[0, 1]
    zoom_range = 400
    mask = ((x_coords > sender_center - zoom_range) & 
            (x_coords < sender_center + zoom_range) &
            (y_coords > total_length/2 - zoom_range) & 
            (y_coords < total_length/2 + zoom_range))
    ax.plot(x_coords[mask], y_coords[mask], 'k.', markersize=2)
    
    # Refinement region
    refinement_rect = plt.Rectangle(
        (sender_center - refinement_region_size/2, total_length/2 - refinement_region_size/2),
        refinement_region_size, refinement_region_size,
        fill=False, edgecolor='blue', linewidth=1, linestyle='--'
    )
    ax.add_patch(refinement_rect)
    
    # Node region
    node_rect = plt.Rectangle(
        (sender_center - node_size/2, total_length/2 - node_size/2),
        node_size, node_size,
        fill=False, edgecolor='red', linewidth=2
    )
    ax.add_patch(node_rect)
    
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title(f'Mesh Near Sender (±{zoom_range} μm)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 3. Cell spacing vs position (X direction)
    ax = axes[1, 0]
    x_cell_centers = 0.5 * (x_positions[:-1] + x_positions[1:])
    
    ax.plot(x_cell_centers, dx_array, 'b-', linewidth=1)
    ax.axhline(y=fine_dx, color='g', linestyle='--', label=f'Fine: {fine_dx} μm')
    ax.axhline(y=coarse_dx, color='r', linestyle='--', label=f'Coarse: {coarse_dx} μm')
    
    for center in node_centers_x:
        ax.axvline(x=center, color='k', linestyle=':', alpha=0.5)
        ax.axvspan(center - refinement_region_size/2, center + refinement_region_size/2, 
                   alpha=0.15, color='blue', label='Refinement region' if center == node_centers_x[0] else '')
        ax.axvspan(center - node_size/2, center + node_size/2, 
                   alpha=0.2, color='red')
    
    ax.set_xlabel('X Position (μm)')
    ax.set_ylabel('Cell Spacing dx (μm)')
    ax.set_title('Cell Spacing in X Direction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Cell spacing vs position (Y direction)
    ax = axes[1, 1]
    y_cell_centers = 0.5 * (y_positions[:-1] + y_positions[1:])
    
    ax.plot(y_cell_centers, dy_array, 'b-', linewidth=1)
    ax.axhline(y=fine_dx, color='g', linestyle='--', label=f'Fine: {fine_dx} μm')
    ax.axhline(y=coarse_dx, color='r', linestyle='--', label=f'Coarse: {coarse_dx} μm')
    
    for center in node_centers_y:
        ax.axvline(x=center, color='k', linestyle=':', alpha=0.5)
        ax.axvspan(center - refinement_region_size/2, center + refinement_region_size/2, 
                   alpha=0.15, color='blue')
        ax.axvspan(center - node_size/2, center + node_size/2, 
                   alpha=0.2, color='red')
    
    ax.set_xlabel('Y Position (μm)')
    ax.set_ylabel('Cell Spacing dy (μm)')
    ax.set_title('Cell Spacing in Y Direction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mesh_visualization_2D_localized.png', dpi=300, bbox_inches='tight')
    print("\nMesh visualization saved as 'mesh_visualization_2D_localized.png'")
    plt.show()
    
    return mesh, dx_array, dy_array


# Example usage
if __name__ == "__main__":
    mesh, dx_array, dy_array = create_and_visualize_2D_nonuniform_mesh(
        node_size=50.0,
        sender_center=1000.0,
        receiver_center=2500.0,  # 1500 μm distance
        fine_dx=5.0,
        coarse_dx=50.0,
        refinement_region_size=300.0,  # Refine in 300×300 μm box around each node
        transition_width=100.0,
        total_length=5000.0
    )