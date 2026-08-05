# test_gmsh_mesh_validation.py
import numpy as np
import matplotlib.pyplot as plt
from New_simple_mesh import create_gmsh_radial_mesh

# Create the mesh with YOUR parameters
mesh, sender_x, receiver_x, y_ctr = create_gmsh_radial_mesh(
    bath_width=10000.0,       # 1 cm
    bath_height=1000.0,       # 1 mm  
    node_diameter=75.0,       # 75 μm nodes
    distance_between_nodes=300.0,
    min_cell_size=0.75,       # 0.75 μm at surface
    max_cell_size=50.0,       # 50 μm far away
    growth_rate=1.5,
    mesh_filename='validation_mesh.msh',
    visualize_gmsh=False,
    verbose=True
)

# Get cell positions
x_coords = mesh.cellCenters[0].value
y_coords = mesh.cellCenters[1].value

# Calculate distance from nearest node for each cell
dist_to_sender = np.sqrt((x_coords - sender_x)**2 + (y_coords - y_ctr)**2)
dist_to_receiver = np.sqrt((x_coords - receiver_x)**2 + (y_coords - y_ctr)**2)
dist_to_nearest_node = np.minimum(dist_to_sender, dist_to_receiver)

# Plot triangular mesh structure (like your image)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1: Full view with triangle edges
ax = axes[0]
faceVertexIDs = mesh.faceVertexIDs  # Returns [3, N] array for triangles

for i in range(mesh.numberOfFaces):
    # Get vertex coordinates for this triangle
    v_ids = faceVertexIDs[:, i]
    vertex_coords = mesh.vertexCoords[:, v_ids]
    
    # Close the triangle
    triangle_x = np.append(vertex_coords[0, :], vertex_coords[0, 0])
    triangle_y = np.append(vertex_coords[1, :], vertex_coords[1, 0])
    
    ax.plot(triangle_x, triangle_y, 'k-', linewidth=0.2, alpha=0.4)

# Highlight nodes
node_circle_sender = plt.Circle((sender_x, y_ctr), radius, 
                                color='blue', fill=False, linewidth=2)
node_circle_receiver = plt.Circle((receiver_x, y_ctr), radius, 
                                  color='red', fill=False, linewidth=2)
ax.add_patch(node_circle_sender)
ax.add_patch(node_circle_receiver)

ax.set_xlim([sender_x - 500, receiver_x + 500])
ax.set_ylim([y_ctr - 300, y_ctr + 300])
ax.set_aspect('equal')
ax.set_title('Full Triangular Mesh Structure', fontsize=14, fontweight='bold')
ax.set_xlabel('X Position (μm)')
ax.set_ylabel('Y Position (μm)')

# Panel 2: Cell size distribution
ax = axes[1]
scatter = ax.scatter(x_coords, y_coords, c=dist_to_nearest_node, 
                     s=0.5, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, ax=ax, label='Distance to nearest node (μm)')

# Circle nodes
ax.add_patch(plt.Circle((sender_x, y_ctr), radius, 
                        color='blue', fill=False, linewidth=2))
ax.add_patch(plt.Circle((receiver_x, y_ctr), radius, 
                        color='red', fill=False, linewidth=2))

ax.set_xlim([sender_x - 500, receiver_x + 500])
ax.set_ylim([y_ctr - 300, y_ctr + 300])
ax.set_aspect('equal')
ax.set_title('Cell Density by Distance from Nodes', fontsize=14, fontweight='bold')
ax.set_xlabel('X Position (μm)')
ax.set_ylabel('Y Position (μm)')

plt.tight_layout()
plt.savefig('mesh_validation_visual.png', dpi=300)
print("\n✓ Saved mesh visualization: mesh_validation_visual.png")
plt.show()