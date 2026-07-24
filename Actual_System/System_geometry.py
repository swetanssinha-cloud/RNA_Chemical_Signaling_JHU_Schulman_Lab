import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set up the figure with better proportions
fig, ax = plt.subplots(figsize=(16, 8))

# Parameters (in micrometers)
tube_width = 10000  # 1 cm = 10000 μm
tube_height = 1000  # 1 mm = 1000 μm
center_distance = 600  # μm
node_radius = 75  # μm
margin = 600  # μm (visual only)

# Calculate positions
view_left = -margin
view_right = center_distance + margin
sender_x = 0
sender_y = tube_height / 2
receiver_x = center_distance
receiver_y = tube_height / 2

# Draw horizontal lines for top and bottom of tube (extending beyond view)
line_extension = 1250
ax.plot([view_left - line_extension, view_right + line_extension], 
        [tube_height, tube_height], 'k-', linewidth=2.5, solid_capstyle='butt', clip_on=False)
ax.plot([view_left - line_extension, view_right + line_extension], 
        [0, 0], 'k-', linewidth=2.5, solid_capstyle='butt', clip_on=False)

# Draw sender node (blue circle)
sender_circle = patches.Circle((sender_x, sender_y), node_radius, 
                               color='lightblue', ec='blue', linewidth=2.5, zorder=3)
ax.add_patch(sender_circle)
ax.plot(sender_x, sender_y, 'ko', markersize=5, zorder=4)

# Draw receiver node (red/pink circle)
receiver_circle = patches.Circle((receiver_x, receiver_y), node_radius, 
                                 color='lightcoral', ec='red', linewidth=2.5, zorder=3)
ax.add_patch(receiver_circle)
ax.plot(receiver_x, receiver_y, 'ko', markersize=5, zorder=4)

# Draw dashed line connecting centers
ax.plot([sender_x, receiver_x], [sender_y, receiver_y], 
        'k--', linewidth=1.5, alpha=0.6, zorder=2)

# Add labels for sender and receiver
ax.text(sender_x, sender_y - 280, 'Sender', 
        ha='center', va='top', fontsize=14, color='blue', fontweight='bold')
ax.text(receiver_x, receiver_y - 280, 'Receiver', 
        ha='center', va='top', fontsize=14, color='red', fontweight='bold')

# Add dimension arrows and labels
# Center distance arrow (higher up to avoid collision)
# Add dimension arrows and labels
# Center distance arrow (higher up to avoid collision)
arrow_y = sender_y
ax.annotate('', xy=(receiver_x, arrow_y), xytext=(sender_x, arrow_y),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
ax.text((sender_x + receiver_x) / 2, arrow_y - 80, 
        '"x"', 
        ha='center', va='top', fontsize=11, bbox=dict(boxstyle='round,pad=0.5', 
        facecolor='white', edgecolor='gray', alpha=0.8))

# Sender radius label
ax.text(sender_x - node_radius - 180, sender_y + 150, 
        f'radius\n= {node_radius} μm', 
        ha='right', va='center', fontsize=10, color='blue',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.3))

# Receiver radius label
ax.text(receiver_x + node_radius + 180, receiver_y + 150, 
        f'radius\n= {node_radius} μm', 
        ha='left', va='center', fontsize=10, color='red',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', alpha=0.3))

# Tube height annotation (on the left side)
tube_height_x = view_left - 350
ax.annotate('', xy=(tube_height_x, tube_height), xytext=(tube_height_x, 0),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
ax.text(tube_height_x - 80, tube_height / 2, 
        f'Tube height\n= 1 mm\n(1000 μm)', 
        ha='center', va='center', fontsize=10, rotation=90,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.5))

# Add dots to indicate continuation (moved outside the main plot area)
dot_y = tube_height / 2
ax.text(view_left - 1100, dot_y, '•  •  •', 
        ha='center', va='center', fontsize=18, fontweight='bold', clip_on=False)
ax.text(view_right + 1100, dot_y, '•  •  •', 
        ha='center', va='center', fontsize=18, fontweight='bold', clip_on=False)

# # Add margin annotations at the bottom (more space)
margin_y = -450

# Set axis properties with more room
ax.set_xlim(view_left - 1400, view_right + 1400)
ax.set_ylim(-700, tube_height + 700)
ax.set_aspect('equal')
ax.set_xlabel('x (μm)', fontsize=12, fontweight='bold')
ax.set_ylabel('y (μm)', fontsize=12, fontweight='bold')

# Add title
ax.set_title('Simulation Geometry\n(Not to scale in the vertical direction)', 
             fontsize=16, fontweight='bold', pad=20)

# Add grid for reference
ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)

# Add legend/info box in a better location (top left)
info_text = (
    'Actual dimensions (used in simulation):\n\n'
    f'  • Tube width     = 1 cm  = 10000 μm\n'
    f'  • Tube height    = 1 mm  = 1000 μm\n'
    f'  • Center distance = {center_distance} μm\n'
    f'  • Node radius     = {node_radius} μm\n'
    f'  • Boundry Conditions = Reflective'
)
ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.8, edgecolor='black'))

plt.tight_layout()
plt.savefig('simulation_geometry.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()