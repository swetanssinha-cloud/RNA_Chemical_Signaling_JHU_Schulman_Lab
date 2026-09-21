"""
Schematic (non-mesh) drawing of the N-node transmission-line geometry.

Mirrors Mesh/Geometry_drawing.py's style -- an oversized, labeled schematic
for explaining the setup, NOT the actual FEM mesh (that's already visualized
by Transmission_Line.py's final-timestep spatial heat map section) -- but
generalizes it from a single sender/receiver pair to N nodes in a row.

Geometry parameters are duplicated here (not imported) from
Transmission_Line.py's PARAMETERS section, exactly like Geometry_drawing.py
hardcodes its own copy of the 2-node model's dimensions rather than
importing the simulation script (which would trigger mesh generation and a
full 8-hour run just to draw a diagram). Keep these in sync by hand if you
change N_NODES / distance_between / node_diameter / total_height /
bath_margin in Transmission_Line.py.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- geometry parameters (mirror Transmission_Line.py) ----------------------
N_NODES = 3
distance_between = 300.0    # um, center-to-center
node_diameter = 75.0        # um
node_radius = node_diameter / 2.0
bath_margin = 2350.0
total_height = 5000.0

chain_span = (N_NODES - 1) * distance_between
total_width = chain_span + node_diameter + 2 * bath_margin

# um (visual only, exaggerated for visibility) -- capped relative to
# distance_between so oversized circles never touch/overlap regardless of
# how tight the node spacing is (node_radius*6 alone is only safe for the
# original 2-node model's much larger 1200 um gap).
display_node_radius = min(node_radius * 6, 0.35 * distance_between)

# --- node positions (same formula as create_conformal_line_mesh) -----------
domain_center_x = total_width / 2.0
node_centers_x = [domain_center_x - chain_span / 2.0 + i * distance_between
                   for i in range(N_NODES)]
node_y = total_height / 2.0

fig, ax = plt.subplots(figsize=(4 * N_NODES + 4, 8))

view_left, view_right = 0, total_width

# Draw the four sides of the tube as a closed box
for (x0, x1), (y0, y1) in [((view_left, view_right), (total_height, total_height)),
                            ((view_left, view_right), (0, 0)),
                            ((view_left, view_left), (0, total_height)),
                            ((view_right, view_right), (0, total_height))]:
    ax.plot([x0, x1], [y0, y1], 'k-', linewidth=2.5, solid_capstyle='butt', clip_on=False)

colors = plt.cm.tab10(np.linspace(0, 1, 10))

for i, cx in enumerate(node_centers_x):
    color = colors[i % 10]
    circle = patches.Circle((cx, node_y), display_node_radius,
                             facecolor=color, edgecolor='black', linewidth=2.0,
                             alpha=0.85, zorder=3)
    ax.add_patch(circle)
    # A number inside the circle rather than a "Node k" label underneath --
    # at 300 um node spacing the oversized circles sit close enough together
    # that wider text labels below them collide with their neighbors.
    ax.text(cx, node_y, str(i + 1), ha='center', va='center',
            fontsize=13, fontweight='bold', color='black', zorder=5)

# Dashed line connecting all node centers down the chain
ax.plot(node_centers_x, [node_y] * N_NODES, 'k--', linewidth=1.5, alpha=0.5, zorder=2)

# Spacing arrow + label between every adjacent pair of nodes
arrow_y = node_y
for i in range(N_NODES - 1):
    x0, x1 = node_centers_x[i], node_centers_x[i + 1]
    ax.annotate('', xy=(x1, arrow_y), xytext=(x0, arrow_y),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text((x0 + x1) / 2, arrow_y - 80, f'{distance_between:.0f} um',
            ha='center', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', alpha=0.8))

# Radius callout on the first node only (every node has the same radius, so
# one label plus the info box below is enough -- avoids N repeated labels)
ax.text(node_centers_x[0] - display_node_radius - 180, node_y + 150,
        f'radius\n= {node_radius:.1f} um',
        ha='right', va='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.6))

# Tube height annotation (left side) -- fixed absolute offsets (not scaled
# by total_width) so it clears the info box regardless of domain size
tube_height_x = view_left - 350
ax.annotate('', xy=(tube_height_x, total_height), xytext=(tube_height_x, 0),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
ax.text(tube_height_x - 80, total_height / 2,
        f'Tube height\n= {total_height:.0f} um',
        ha='center', va='center', fontsize=10, rotation=90,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.5))

ax.text((view_left + view_right) / 2, -total_height * 0.06,
        'Nodes numbered 1 to N left-to-right (circles oversized for visibility)\n'
        'Node 1 is a fixed source, not solved',
        ha='center', va='top', fontsize=10, style='italic', color='dimgray')

ax.set_xlim(view_left - 1400, view_right + 1400)
ax.set_ylim(-total_height * 0.22, total_height * 1.15)
ax.set_aspect('equal')
ax.set_xlabel('x (um)', fontsize=12, fontweight='bold')
ax.set_ylabel('y (um)', fontsize=12, fontweight='bold')
ax.set_title(f'{N_NODES}-Node Transmission Line: Simulation Geometry',
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)

info_text = (
    'Actual dimensions (used in simulation):\n\n'
    f'  - Number of nodes = {N_NODES}\n'
    f'  - Tube width      = {total_width:.0f} um\n'
    f'  - Tube height     = {total_height:.0f} um\n'
    f'  - Node spacing    = {distance_between:.0f} um (center-to-center)\n'
    f'  - Node radius     = {node_radius:.1f} um\n'
    f'  - Boundary conditions = Reflective'
)
# Anchored in DATA coordinates (top-right of the tube) rather than axes
# fraction, so it stays clear of the tube-height annotation on the left
# regardless of how total_width scales with N_NODES.
ax.text(total_width * 0.98, total_height * 0.98, info_text,
        ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.8, edgecolor='black'))

plt.tight_layout()
plt.savefig('transmission_line_geometry.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
