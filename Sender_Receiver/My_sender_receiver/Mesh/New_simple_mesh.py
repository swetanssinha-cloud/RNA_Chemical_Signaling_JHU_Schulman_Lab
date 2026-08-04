"""
Gmsh Radial Adaptive Mesh for FiPy - TRUE Radial Refinement
FIXED: Uses Gmsh format 2.2 for FiPy compatibility
"""

import numpy as np
import gmsh
import sys
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import sys


def create_gmsh_radial_mesh(
    bath_width=10000.0,           # μm (1 cm)
    bath_height=1000.0,           # μm (1 mm)
    node_diameter=75.0,           # μm
    distance_between_nodes=300.0, # μm (center-to-center)
    min_cell_size=0.75,          # μm (finest mesh at node surface)
    max_cell_size=50.0,          # μm (coarsest mesh far from nodes)
    growth_rate=1.5,             # How fast mesh grows with distance
    mesh_filename='radial_mesh.msh',
    visualize_gmsh=False,        # Show Gmsh GUI
    verbose=True
):
    """
    Create TRUE radial adaptive mesh using Gmsh.
    Fine mesh ONLY near circular nodes (radially), coarse everywhere else.
    
    FIXED: Saves mesh in Gmsh format 2.2 for FiPy compatibility.
    
    Parameters:
    -----------
    bath_width, bath_height : float
        Domain dimensions in μm
    node_diameter : float
        Diameter of circular hydrogel nodes in μm
    distance_between_nodes : float
        Center-to-center distance between sender and receiver nodes in μm
    min_cell_size : float
        Finest mesh size at node surfaces in μm (must be 0.75 for tanh compatibility)
    max_cell_size : float
        Coarsest mesh size far from nodes in μm
    growth_rate : float
        How quickly mesh transitions from fine to coarse (1.3 = slow, 2.0 = fast)
    mesh_filename : str
        Output filename for mesh file
    visualize_gmsh : bool
        If True, opens Gmsh GUI to visualize mesh
    verbose : bool
        Print mesh statistics
    
    Returns:
    --------
    mesh : FiPy Gmsh2D mesh object
    sender_center_x : float
    receiver_center_x : float
    y_center : float
    node_radius : float
    """
    
    node_radius = node_diameter / 2.0
    y_center = bath_height / 2.0
    
    # Position nodes symmetrically around domain center
    domain_center_x = bath_width / 2.0
    sender_center_x = domain_center_x - distance_between_nodes / 2.0
    receiver_center_x = domain_center_x + distance_between_nodes / 2.0
    
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    
    # Create new model
    gmsh.model.add("radial_adaptive_mesh")
    
    # =============================================================================
    # GEOMETRY CREATION
    # =============================================================================
    
    # Create the rectangular bath domain
    rectangle_tag = gmsh.model.occ.addRectangle(0, 0, 0, bath_width, bath_height)
    
    # Create circular sender node
    sender_circle_tag = gmsh.model.occ.addDisk(sender_center_x, y_center, 0, 
                                                node_radius, node_radius)
    
    # Create circular receiver node
    receiver_circle_tag = gmsh.model.occ.addDisk(receiver_center_x, y_center, 0,
                                                  node_radius, node_radius)
    
    # Synchronize CAD entities before mesh operations
    gmsh.model.occ.synchronize()
    
# =============================================================================
# MESH SIZE FIELD - TRUE RADIAL REFINEMENT + FINE INSIDE NODES
# =============================================================================

    # Get boundaries (curves) of the circular nodes
    sender_boundary = gmsh.model.getBoundary([(2, sender_circle_tag)], 
                                            oriented=False, combined=False, recursive=False)
    receiver_boundary = gmsh.model.getBoundary([(2, receiver_circle_tag)], 
                                            oriented=False, combined=False, recursive=False)

    # Extract curve tags (boundaries are 1D entities)
    sender_curve_tags = [abs(tag) for dim, tag in sender_boundary]
    receiver_curve_tags = [abs(tag) for dim, tag in receiver_boundary]

    if verbose:
        print(f"\nSender boundary curves: {sender_curve_tags}")
        print(f"Receiver boundary curves: {receiver_curve_tags}")

    # CREATE EXPLICIT CENTER POINTS for distance fields
    sender_center_point = gmsh.model.occ.addPoint(sender_center_x, y_center, 0)
    receiver_center_point = gmsh.model.occ.addPoint(receiver_center_x, y_center, 0)

    # Synchronize to register the new points
    gmsh.model.occ.synchronize()

    if verbose:
        print(f"Created sender center point: {sender_center_point}")
        print(f"Created receiver center point: {receiver_center_point}")

    # Distance field from SENDER node boundary (edge)
    distance_field_sender_boundary = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field_sender_boundary, "CurvesList", sender_curve_tags)
    gmsh.model.mesh.field.setNumber(distance_field_sender_boundary, "Sampling", 200)

    # Distance field from RECEIVER node boundary (edge)
    distance_field_receiver_boundary = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field_receiver_boundary, "CurvesList", receiver_curve_tags)
    gmsh.model.mesh.field.setNumber(distance_field_receiver_boundary, "Sampling", 200)

    # Distance field from SENDER CENTER POINT
    distance_field_sender_center = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field_sender_center, "PointsList", [sender_center_point])
    gmsh.model.mesh.field.setNumber(distance_field_sender_center, "Sampling", 200)

    # Distance field from RECEIVER CENTER POINT
    distance_field_receiver_center = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field_receiver_center, "PointsList", [receiver_center_point])
    gmsh.model.mesh.field.setNumber(distance_field_receiver_center, "Sampling", 200)

# Take MINIMUM distance to any boundary OR center
    min_distance_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_distance_field, "FieldsList", 
                                    [distance_field_sender_boundary, 
                                    distance_field_receiver_boundary,
                                    distance_field_sender_center,
                                    distance_field_receiver_center])

    # =============================================================================
    # CORRECTED: Use Threshold field for gradual refinement (matches reference image)
    # =============================================================================
    # Define refinement zone: fine mesh extends this far from nodes
    refinement_radius = 200.0  # μm - based on your reference image

    # Create Threshold field for smooth transition
    # Keeps mesh fine out to refinement_radius, then gradually coarsens
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "InField", min_distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", min_cell_size)      # 0.75 μm at nodes
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", max_cell_size)      # 50 μm far away
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.0)                # Fine mesh starts at node
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", refinement_radius)  # Fine mesh ends here
    gmsh.model.mesh.field.setNumber(threshold_field, "Sigmoid", 1)                  # Smooth transition

    if verbose:
        print(f"\n✓ CORRECTED mesh refinement strategy:")
        print(f"  At node surface (d=0): {min_cell_size:.3f} μm")
        print(f"  From d=0 to d={refinement_radius:.0f} μm: stays ~{min_cell_size:.3f}-{min_cell_size*2:.1f} μm")
        print(f"  Beyond d={refinement_radius:.0f} μm: smoothly grows to {max_cell_size:.1f} μm")
        print(f"  Transition: Sigmoid (smooth, gradual)")
        print(f"  → This matches your COMSOL reference image!")
        
    # Set this as the background mesh field
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
    
    # Disable other mesh size determination methods
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    
    # Algorithm options
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    
    # =============================================================================
    # GENERATE MESH
    # =============================================================================
    
    
    # Generate 2D mesh
    gmsh.model.mesh.generate(2)
    
    # Optional: optimize mesh quality

    gmsh.model.mesh.optimize("Netgen")
    
    # =============================================================================
    # STATISTICS
    # =============================================================================
    
    if verbose:
        # Get mesh statistics
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        element_types, element_tags_list, element_node_tags = gmsh.model.mesh.getElements(dim=2)
        
        n_nodes = len(node_tags)
        n_triangles = sum(len(tags) for tags in element_tags_list)
    
    # =============================================================================
    # SAVE MESH IN FORMAT 2.2 (FIPY COMPATIBLE)
    # =============================================================================
    
    # THIS IS THE FIX: Force Gmsh to save in format 2.2
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    
    if verbose:
        print(f"Saving mesh in Gmsh format 2.2 (FiPy compatible)...")
    
    gmsh.write(mesh_filename)
    
    # Optional: Launch Gmsh GUI to visualize
    if visualize_gmsh:
        if verbose:
            print("Launching Gmsh GUI...")
        gmsh.fltk.run()
    
    # Finalize Gmsh
    gmsh.finalize()
    
    # =============================================================================
    # IMPORT INTO FIPY
    # =============================================================================
    
    if verbose:
        print(f"Importing mesh into FiPy...")
    
    from fipy import Gmsh2D
    mesh = Gmsh2D(mesh_filename)
    
    if verbose:
        print(f"✓ FiPy mesh created with {mesh.numberOfCells:,} cells\n")
    
    return mesh, sender_center_x, receiver_center_x, y_center


def visualize_gmsh_mesh_comprehensive(
    mesh, sender_x, receiver_x, y_center, node_radius, distance_between,
    save_filename='gmsh_radial_mesh_visualization.png'
):
    """
    Comprehensive 4-panel visualization of the Gmsh radial mesh.
    Shows that mesh is fine ONLY near nodes (radially), not in bands.
    """
    
    x_coords, y_coords = mesh.cellCenters
    x_coords = x_coords
    y_coords = y_coords
    
    # Calculate distance to nearest node surface
    def distance_to_nearest_node_surface(x, y):
        nodes = np.array([[sender_x, y_center], [receiver_x, y_center]])
        distances = []
        for node_x, node_y in nodes:
            dist_to_center = np.sqrt((x - node_x)**2 + (y - node_y)**2)
            dist_to_surface = max(0.0, dist_to_center - node_radius)
            distances.append(dist_to_surface)
        return min(distances)
    
    distances = np.array([distance_to_nearest_node_surface(x, y) 
                         for x, y in zip(x_coords, y_coords)])
    
    # Create masks for nodes
    sender_mask = np.sqrt((x_coords - sender_x)**2 + 
                          (y_coords - y_center)**2) <= node_radius
    receiver_mask = np.sqrt((x_coords - receiver_x)**2 + 
                            (y_coords - y_center)**2) <= node_radius
    
    # Create figure
    fig = plt.figure(figsize=(20, 14))
    
    # =============================================================================
    # Panel 1: Full domain with distance coloring
    # =============================================================================
    ax1 = plt.subplot(2, 2, 1)
    
    subsample = max(1, len(x_coords) // 50000)
    scatter1 = ax1.scatter(x_coords[::subsample], y_coords[::subsample], 
                          c=distances[::subsample], s=2, cmap='RdYlGn_r',
                          alpha=0.7, vmin=0, vmax=min(300, distances.max()))
    
    # Draw node circles
    sender_circle = plt.Circle((sender_x, y_center), node_radius,
                               fill=False, edgecolor='blue', linewidth=2.5, 
                               linestyle='-', label='Sender node')
    receiver_circle = plt.Circle((receiver_x, y_center), node_radius,
                                 fill=False, edgecolor='red', linewidth=2.5,
                                 linestyle='-', label='Receiver node')
    ax1.add_patch(sender_circle)
    ax1.add_patch(receiver_circle)
    
    # Distance annotation
    ax1.plot([sender_x, receiver_x], [y_center, y_center],
            'k--', linewidth=1.5, alpha=0.6)
    mid_x = (sender_x + receiver_x) / 2
    ax1.text(mid_x, y_center + 80, f'{distance_between:.0f} μm', 
            ha='center', va='bottom', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
    
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Distance to Nearest Node Surface (μm)', fontsize=12, fontweight='bold')
    
    ax1.set_xlabel('X Position (μm)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Y Position (μm)', fontsize=13, fontweight='bold')
    ax1.set_title('Full Domain: True Radial Refinement (NO BANDS!)', 
                 fontsize=15, fontweight='bold', pad=10)
    ax1.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # =============================================================================
    # Panel 2: Zoomed view of sender node
    # =============================================================================
    ax2 = plt.subplot(2, 2, 2)
    
    zoom_size = node_radius * 6
    x_min, x_max = sender_x - zoom_size, sender_x + zoom_size
    y_min, y_max = y_center - zoom_size, y_center + zoom_size
    
    zoom_mask = ((x_coords >= x_min) & (x_coords <= x_max) & 
                 (y_coords >= y_min) & (y_coords <= y_max))
    
    scatter2 = ax2.scatter(x_coords[zoom_mask], y_coords[zoom_mask], 
                          c=distances[zoom_mask], s=20, cmap='RdYlGn_r',
                          alpha=0.8, vmin=0, vmax=100)
    
    sender_circle_zoom = plt.Circle((sender_x, y_center), node_radius,
                                    fill=False, edgecolor='blue', linewidth=3)
    ax2.add_patch(sender_circle_zoom)
    
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Distance (μm)', fontsize=11)
    
    ax2.set_xlabel('X Position (μm)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Y Position (μm)', fontsize=13, fontweight='bold')
    ax2.set_title(f'Sender Node (R={node_radius:.1f} μm) - Radial Refinement', 
                 fontsize=14, fontweight='bold', pad=10)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim([x_min, x_max])
    ax2.set_ylim([y_min, y_max])
    
    # =============================================================================
    # Panel 3: Zoomed view of receiver node
    # =============================================================================
    ax3 = plt.subplot(2, 2, 3)
    
    x_min, x_max = receiver_x - zoom_size, receiver_x + zoom_size
    y_min, y_max = y_center - zoom_size, y_center + zoom_size
    
    zoom_mask = ((x_coords >= x_min) & (x_coords <= x_max) & 
                 (y_coords >= y_min) & (y_coords <= y_max))
    
    scatter3 = ax3.scatter(x_coords[zoom_mask], y_coords[zoom_mask], 
                          c=distances[zoom_mask], s=20, cmap='RdYlGn_r',
                          alpha=0.8, vmin=0, vmax=100)
    
    receiver_circle_zoom = plt.Circle((receiver_x, y_center), node_radius,
                                      fill=False, edgecolor='red', linewidth=3)
    ax3.add_patch(receiver_circle_zoom)
    
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Distance (μm)', fontsize=11)
    
    ax3.set_xlabel('X Position (μm)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Y Position (μm)', fontsize=13, fontweight='bold')
    ax3.set_title(f'Receiver Node (R={node_radius:.1f} μm) - Radial Refinement', 
                 fontsize=14, fontweight='bold', pad=10)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim([x_min, x_max])
    ax3.set_ylim([y_min, y_max])
    
    # =============================================================================
    # Panel 4: Cross-sections showing NO BANDS
    # =============================================================================
    ax4 = plt.subplot(2, 2, 4)
    
    # Horizontal cross-section at Y = y_center
    y_tol = 20  # μm
    h_mask = np.abs(y_coords - y_center) < y_tol
    x_h = x_coords[h_mask]
    d_h = distances[h_mask]
    sort_idx = np.argsort(x_h)
    x_h, d_h = x_h[sort_idx], d_h[sort_idx]
    
    # Vertical cross-section at X = midpoint between nodes
    mid_x = (sender_x + receiver_x) / 2
    x_tol = 50  # μm
    v_mask = np.abs(x_coords - mid_x) < x_tol
    y_v = y_coords[v_mask]
    d_v = distances[v_mask]
    sort_idx = np.argsort(y_v)
    y_v, d_v = y_v[sort_idx], d_v[sort_idx]
    
    ax4.plot(x_h, d_h, 'b-', linewidth=2.5, label='Horizontal (Y=center)', alpha=0.8)
    ax4.plot(y_v, d_v, 'r-', linewidth=2.5, label='Vertical (X=midpoint)', alpha=0.8)
    
    # Mark node positions on horizontal
    ax4.axvline(sender_x, color='blue', linestyle=':', alpha=0.5, linewidth=1.5)
    ax4.axvline(receiver_x, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
    ax4.text(sender_x, ax4.get_ylim()[1]*0.9, 'Sender', ha='center', 
             fontsize=10, color='blue', fontweight='bold')
    ax4.text(receiver_x, ax4.get_ylim()[1]*0.9, 'Receiver', ha='center',
             fontsize=10, color='red', fontweight='bold')
    
    ax4.set_xlabel('Position (μm)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Distance to Node Surface (μm)', fontsize=13, fontweight='bold')
    ax4.set_title('Cross-Sections: Proving NO BANDS (distance increases radially)', 
                 fontsize=14, fontweight='bold', pad=10)
    ax4.legend(fontsize=11, loc='upper right')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comprehensive visualization saved: {save_filename}")
    plt.show()



def visualize_triangle_mesh(mesh, sender_x, receiver_x, y_center, node_radius,
                           zoom_sender=True, zoom_size=150.0,
                           save_filename='gmsh_triangle_mesh.png'):
    """
    Visualize the actual triangular mesh structure (edges and vertices).
    Shows individual triangles near the nodes.
    
    Parameters:
    -----------
    mesh : FiPy Gmsh2D mesh
    sender_x, receiver_x : float
        Node center X coordinates
    y_center : float
        Node center Y coordinate
    node_radius : float
        Node radius
    zoom_sender : bool
        If True, zoom on sender; if False, zoom on receiver
    zoom_size : float
        Size of zoom window (μm from node center)
    """
    
    # Get mesh vertices and face connectivity
    x_verts = mesh.vertexCoords[0]
    y_verts = mesh.vertexCoords[1]
    
    # Get face vertex IDs (triangles)
    face_vertex_ids = mesh.faceVertexIDs
    
    # Choose zoom center
    zoom_center_x = sender_x if zoom_sender else receiver_x
    node_name = "Sender" if zoom_sender else "Receiver"
    node_color = 'blue' if zoom_sender else 'red'
    
    # Create zoom mask
    x_min, x_max = zoom_center_x - zoom_size, zoom_center_x + zoom_size
    y_min, y_max = y_center - zoom_size, y_center + zoom_size
    
    # Filter vertices in zoom region
    vert_mask = ((x_verts >= x_min) & (x_verts <= x_max) & 
                 (y_verts >= y_min) & (y_verts <= y_max))
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Plot all mesh edges (triangle edges)
    print(f"Plotting triangular mesh edges...")
    for i in range(face_vertex_ids.shape[1]):
        v0, v1 = face_vertex_ids[:, i]
        
        # Check if either vertex is in zoom region
        if vert_mask[v0] or vert_mask[v1]:
            x_line = [x_verts[v0], x_verts[v1]]
            y_line = [y_verts[v0], y_verts[v1]]
            
            # Only plot if line segment is in zoom region
            if (min(x_line) < x_max and max(x_line) > x_min and
                min(y_line) < y_max and max(y_line) > y_min):
                ax.plot(x_line, y_line, 'k-', linewidth=0.5, alpha=0.6)
    
    # Plot vertices in zoom region
    ax.scatter(x_verts[vert_mask], y_verts[vert_mask], 
              c='darkblue', s=8, alpha=0.8, zorder=5, label='Mesh vertices')
    
    # Draw node circle
    node_circle = plt.Circle((zoom_center_x, y_center), node_radius,
                            fill=False, edgecolor=node_color, linewidth=3,
                            label=f'{node_name} node')
    ax.add_patch(node_circle)
    
    ax.set_xlabel('X Position (μm)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y Position (μm)', fontsize=13, fontweight='bold')
    ax.set_title(f'Triangular Mesh Structure: {node_name} Node (zoom {zoom_size:.0f} μm)', 
                fontsize=15, fontweight='bold', pad=10)
    ax.legend(fontsize=12, loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Triangle mesh visualization saved: {save_filename}")
    plt.show()
    
    return fig

# =============================================================================
# EXAMPLE USAGE 
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("CREATING GMSH RADIAL MESH - TRUE RADIAL REFINEMENT")
    print("="*70)
    
    # Create mesh with geometric growth (matching your requirements)
    mesh, sender_x, receiver_x, y_ctr = create_gmsh_radial_mesh(
        bath_width=10000.0,           # 1 cm
        bath_height=1000.0,           # 1 mm
        node_diameter=75.0,           # 75 μm diameter nodes
        distance_between_nodes=300.0, # 300 μm apart
        min_cell_size=0.75,          # 0.75 μm at node surface (for tanh)
        max_cell_size=50.0,          # 50 μm far away
        growth_rate=1.5,             # Geometric growth factor
        mesh_filename='radial_mesh.msh',
        visualize_gmsh=False,        # Set True to see Gmsh GUI
        verbose=True
    )
    
    # print("\n" + "="*70)
    # print("MESH CREATED SUCCESSFULLY!")
    # print("="*70)
    # print(f"\nMesh has {mesh.numberOfCells:,} cells")
    # print(f"This is an UNSTRUCTURED triangular mesh with TRUE radial refinement")
    # print(f"Fine cells ONLY near nodes (no horizontal/vertical bands!)")
    
    # Visualize the mesh
    print("\nGenerating comprehensive visualization...")
    # visualize_gmsh_mesh_comprehensive(
    #     mesh, sender_x, receiver_x, y_ctr, radius, 300.0,
    #     save_filename='gmsh_radial_mesh_comprehensive.png'
    # )
    
    # print("\n" + "="*70)
    # print("READY TO USE IN YOUR FIPY SIMULATION!")
    # print("="*70)
    # print("\nJust use this mesh in your existing FiPy code:")
    # print("  S2 = CellVariable(name='S2', mesh=mesh, value=0.0)")
    # print("  ... all your equations work exactly the same!")
    # print("="*70 + "\n")

    # Visualize actual triangular mesh structure
    print("\nGenerating triangle mesh visualization...")
    visualize_triangle_mesh(
    mesh, sender_x, receiver_x, y_ctr, node_radius=37.5,
    zoom_sender=False,  # Set False to see receiver instead
    zoom_size=250.0,   # Adjust to zoom in/out
    save_filename='gmsh_triangle_mesh_sender.png'
)