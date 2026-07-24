
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# -----------------------------
# Parameters
# -----------------------------
center_distance_um = 3000
tube_length_um = 10000
tube_height_um = 1000

sys_centerx = tube_length_um/2
sys_centery = tube_height_um/2

sender_center_x = sys_centerx - center_distance_um/2
receiver_center_x = sys_centerx + center_distance_um/2
sender_center_y = receiver_center_y = sys_centery

node_radius = 75

# Visualization only
visual_height = 300
tube_top = sys_centery + visual_height/2
tube_bottom = sys_centery - visual_height/2
tube_left = sender_center_x - 250
tube_right = receiver_center_x + 250

fig, ax = plt.subplots(figsize=(12,5))

# Channel walls (open-ended)
ax.plot([tube_left,tube_right],[tube_top,tube_top],lw=2.5,color='black')
ax.plot([tube_left,tube_right],[tube_bottom,tube_bottom],lw=2.5,color='black')
ax.text(tube_left-40,sys_centery,"...",fontsize=22,va='center',ha='right')
ax.text(tube_right+40,sys_centery,"...",fontsize=22,va='center',ha='left')

# Nodes
for x,c,e in [(sender_center_x,'cornflowerblue','blue'),
              (receiver_center_x,'salmon','red')]:
    ax.add_patch(Circle((x,sys_centery),node_radius,facecolor=c,edgecolor=e,alpha=.4,lw=2))
    ax.plot(x,sys_centery,'ko',ms=4)

ax.plot([sender_center_x,receiver_center_x],[sys_centery,sys_centery],'k--')

# Distance
arrow_y=sys_centery+105
ax.annotate("",(receiver_center_x,arrow_y),(sender_center_x,arrow_y),
            arrowprops=dict(arrowstyle="<->",lw=1.8))
ax.text((sender_center_x+receiver_center_x)/2,arrow_y+18,
        "Center distance = 3000 μm",ha='center',
        bbox=dict(facecolor='white',edgecolor='none'))

# Labels
ax.text(sender_center_x,sys_centery-115,"Sender",color='blue',ha='center',fontweight='bold')
ax.text(receiver_center_x,sys_centery-115,"Receiver",color='red',ha='center',fontweight='bold')

# Radius callouts
ax.annotate("r = 75 μm",xy=(sender_center_x+node_radius,sys_centery+10),
            xytext=(sender_center_x-220,sys_centery+95),
            arrowprops=dict(arrowstyle='->',color='blue'),color='blue')
ax.annotate("r = 75 μm",xy=(receiver_center_x+node_radius,sys_centery+10),
            xytext=(receiver_center_x+120,sys_centery+95),
            arrowprops=dict(arrowstyle='->',color='red'),color='red')

# Height
ax.annotate("",(tube_left-120,tube_top),(tube_left-120,tube_bottom),
            arrowprops=dict(arrowstyle="<->"))
ax.text(tube_left-300,sys_centery,"Tube height\n1000 μm",ha='center',va='center')
ax.text(tube_left-300,tube_bottom-35,"(vertical scale compressed)",
        ha='center',fontsize=9,style='italic')

# Parameter box
param_text = (
"Simulation Parameters\n\n"
"Tube length = 10,000 μm (1 cm)\n"
"Tube height = 1,000 μm (1 mm)\n"
"Node radius = 75 μm\n"
"Center spacing = 3,000 μm"
)
ax.text(0.02,0.03,param_text,transform=ax.transAxes,fontsize=10,
        bbox=dict(boxstyle="round",facecolor="white"))

legend = [
    Line2D([0],[0],marker='o',color='w',markerfacecolor='cornflowerblue',markeredgecolor='blue',markersize=10,label='Sender'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor='salmon',markeredgecolor='red',markersize=10,label='Receiver')
]
ax.legend(handles=legend,loc='lower right')

ax.set_xlim(sender_center_x-400,receiver_center_x+400)
ax.set_ylim(sys_centery-200,sys_centery+200)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Simulation Geometry")
plt.tight_layout()
plt.show()

# with open('/mnt/data/simulation_geometry_schematic.py','w') as f:
#     f.write(code)
