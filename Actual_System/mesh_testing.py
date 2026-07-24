import numpy as np
import matplotlib.pyplot as plt
from fipy import Grid2D

def create_and_visualize_2D_nonuniform_mesh(
    node_size=50.0,
    sender_center=1000.0,
    receiver_center=2500.0,
    fine_dx=5.0,
    coarse_dx=50.0,
    box_padding=200.0,  # Padding around the bounding box of both nodes
    transition_width=100.0,
    total_length=5000.0
):
    """
    Create and visualize a 2D non-uniform mesh with a single refinement box
    covering both nodes.
    
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
    total_length : float
        Total domain size (μm)
    """
    
    # Calculate bounding box that encompasses both nodes
    node_centers_x = [sender_center, receiver_center]
    node_centers_y = [total_length/2]  # Center vertically
    
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
        dx = max(x_min - x, 0, x - x_max)
        dy = max(y_min - y, 0, y - y_max)
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
            dx = fine_dx + (coarse_dx - fine_dx) * blend
            
            current_pos += dx
            if current_pos < total_length:
                positions.append(current_pos)
        
        # Ensure we end at total_length
        if positions[-1] < total_length:
            positions.append(total_length)
        
        positions = np.array(positions)
        dx_array = np.diff(positions)
        return positions, dx_array
    
    # First pass: create preliminary Y spacing
    y_positions_prelim = np.linspace(0, total_length, 100)
    
    # Create X spacing considering all Y positions
    x_positions, dx_array = create_adaptive_spacing_1D(
        total_length, y_positions_prelim,
        fine_dx, coarse_dx, is_x_direction=True
    )
    
    # Create Y spacing considering all X positions
    y_positions, dy_array = create_adaptive_spacing_1D(
        total_length, x_positions,
        fine_dx, coarse_dx, is_x_direction=False
    )
    
    # Create the mesh
    mesh = Grid2D(dx=dx_array, dy=dy_array)
    
    # Calculate refinement box dimensions
    box_width = refinement_x_max - refinement_x_min
    box_height = refinement_y_max - refinement_y_min
    
    print(f"Mesh created:")
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
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Full mesh structure
    ax = axes[0, 0]
    x_coords = mesh.cellCenters[0].value
    y_coords = mesh.cellCenters[1].value
    ax.plot(x_coords, y_coords, 'k.', markersize=0.5, alpha=0.3)
    
    # Draw refinement box
    refinement_rect = plt.Rectangle(
        (refinement_x_min, refinement_y_min),
        box_width, box_height,
        fill=False, edgecolor='blue', linewidth=2, linestyle='--',
        label='Refinement box'
    )
    ax.add_patch(refinement_rect)
    
    # Mark node regions
    for nx in node_centers_x:
        node_rect = plt.Rectangle(
            (nx - node_size/2, y_center - node_size/2),
            node_size, node_size,
            fill=True, facecolor='red', edgecolor='darkred', 
            linewidth=2, alpha=0.3,
            label='Node' if nx == node_centers_x[0] else ''
        )
        ax.add_patch(node_rect)
    
    ax.set_xlabel('X (μm)', fontsize=12)
    ax.set_ylabel('Y (μm)', fontsize=12)
    ax.set_title('Full Mesh Structure (cell centers)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # 2. Zoomed view around both nodes
    ax = axes[0, 1]
    zoom_margin = 300
    zoom_x_min = refinement_x_min - zoom_margin
    zoom_x_max = refinement_x_max + zoom_margin
    zoom_y_min = refinement_y_min - zoom_margin
    zoom_y_max = refinement_y_max + zoom_margin
    
    mask = ((x_coords > zoom_x_min) & (x_coords < zoom_x_max) &
            (y_coords > zoom_y_min) & (y_coords < zoom_y_max))
    ax.plot(x_coords[mask], y_coords[mask], 'k.', markersize=1.5)
    
    # Refinement box
    refinement_rect = plt.Rectangle(
        (refinement_x_min, refinement_y_min),
        box_width, box_height,
        fill=False, edgecolor='blue', linewidth=2, linestyle='--',
        label='Refinement box'
    )
    ax.add_patch(refinement_rect)
    
    # Node regions
    for i, nx in enumerate(node_centers_x):
        node_rect = plt.Rectangle(
            (nx - node_size/2, y_center - node_size/2),
            node_size, node_size,
            fill=True, facecolor='red', edgecolor='darkred', 
            linewidth=2, alpha=0.3
        )
        ax.add_patch(node_rect)
        ax.text(nx, y_center, f'N{i+1}', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.7))
    
    ax.set_xlabel('X (μm)', fontsize=12)
    ax.set_ylabel('Y (μm)', fontsize=12)
    ax.set_title('Zoomed View: Refinement Region', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # 3. Cell spacing vs position (X direction)
    ax = axes[1, 0]
    x_cell_centers = 0.5 * (x_positions[:-1] + x_positions[1:])
    
    ax.plot(x_cell_centers, dx_array, 'b-', linewidth=1.5, label='Actual spacing')
    ax.axhline(y=fine_dx, color='g', linestyle='--', linewidth=2, 
               label=f'Fine: {fine_dx} μm')
    ax.axhline(y=coarse_dx, color='r', linestyle='--', linewidth=2,
               label=f'Coarse: {coarse_dx} μm')
    
    # Mark refinement box
    ax.axvspan(refinement_x_min, refinement_x_max, 
               alpha=0.15, color='blue', label='Refinement box')
    
    # Mark nodes
    for i, center in enumerate(node_centers_x):
        ax.axvline(x=center, color='darkred', linestyle=':', linewidth=2, alpha=0.7)
        ax.axvspan(center - node_size/2, center + node_size/2, 
                   alpha=0.2, color='red')
    
    ax.set_xlabel('X Position (μm)', fontsize=12)
    ax.set_ylabel('Cell Spacing dx (μm)', fontsize=12)
    ax.set_title('Cell Spacing in X Direction', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, coarse_dx * 1.1])
    
    # 4. Cell spacing vs position (Y direction)
    ax = axes[1, 1]
    y_cell_centers = 0.5 * (y_positions[:-1] + y_positions[1:])
    
    ax.plot(y_cell_centers, dy_array, 'b-', linewidth=1.5, label='Actual spacing')
    ax.axhline(y=fine_dx, color='g', linestyle='--', linewidth=2,
               label=f'Fine: {fine_dx} μm')
    ax.axhline(y=coarse_dx, color='r', linestyle='--', linewidth=2,
               label=f'Coarse: {coarse_dx} μm')
    
    # Mark refinement box
    ax.axvspan(refinement_y_min, refinement_y_max, 
               alpha=0.15, color='blue', label='Refinement box')
    
    # Mark node center
    ax.axvline(x=y_center, color='darkred', linestyle=':', linewidth=2, alpha=0.7)
    ax.axvspan(y_center - node_size/2, y_center + node_size/2, 
               alpha=0.2, color='red')
    
    ax.set_xlabel('Y Position (μm)', fontsize=12)
    ax.set_ylabel('Cell Spacing dy (μm)', fontsize=12)
    ax.set_title('Cell Spacing in Y Direction', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, coarse_dx * 1.1])
    
    plt.tight_layout()
    plt.savefig('mesh_visualization_2D_unified_box.png', dpi=300, bbox_inches='tight')
    print("\nMesh visualization saved as 'mesh_visualization_2D_unified_box.png'")
    plt.show()
    
    return mesh, dx_array, dy_array, (refinement_x_min, refinement_x_max, 
                                       refinement_y_min, refinement_y_max)


# Example usage
if __name__ == "__main__":
    mesh, dx_array, dy_array, refinement_box = create_and_visualize_2D_nonuniform_mesh(
        node_size=50.0,
        sender_center=1000.0,
        receiver_center=2500.0,  # 1500 μm distance
        fine_dx=5.0,
        coarse_dx=50.0,
        box_padding=200.0,  # 200 μm padding around nodes
        transition_width=100.0,
        total_length=5000.0
    )
    
    print(f"\nRefinement box coordinates returned:")
    print(f"  (x_min, x_max, y_min, y_max) = {refinement_box}")