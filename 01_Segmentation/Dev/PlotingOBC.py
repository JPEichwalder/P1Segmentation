import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for some Matplotlib versions

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# -------------------------
# AXES: positive and negative
# -------------------------
ax.quiver(0, 0, 0,  1.5, 0,   0, color='k', arrow_length_ratio=0.08)
ax.quiver(0, 0, 0, -1.5, 0,   0, color='k', arrow_length_ratio=0.08)

ax.quiver(0, 0, 0,  0,  1.5,  0, color='k', arrow_length_ratio=0.08)
ax.quiver(0, 0, 0,  0, -1.5,  0, color='k', arrow_length_ratio=0.08)

ax.quiver(0, 0, 0,  0,  0,  1.5, color='k', arrow_length_ratio=0.08)
ax.quiver(0, 0, 0,  0,  0, -1.5, color='k', arrow_length_ratio=0.08)

# -------------------------
# AXIS LABELS: moved outward + white background
# -------------------------
label_box = dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5)

ax.text( 1.85, -0.10, -0.08, 'x = Density',
         fontsize=10, ha='left', va='center', bbox=label_box)

ax.text(-1.85,  0.05,  0.00, '-x',
         fontsize=10, ha='right', va='center', bbox=label_box)

ax.text( 0.10,  1.85,  0.02, 'y = Neighbouring Context',
         fontsize=10, ha='left', va='center', bbox=label_box)

ax.text(-0.05, -1.85,  0.00, '-y',
         fontsize=10, ha='right', va='center', bbox=label_box)

ax.text( 0.00,  0.05,  1.85, 'z = Geometry',
         fontsize=10, ha='center', va='bottom', bbox=label_box)

ax.text( 0.00,  0.00, -1.90, '-z',
         fontsize=10, ha='center', va='top', bbox=label_box)

# -------------------------
# 3 COLORED VECTORS = voxel vectors
# -------------------------
ax.quiver(0, 0, 0,  1,  1,  1, color='blue',       linewidth=1.5, arrow_length_ratio=0.10)
ax.quiver(0, 0, 0, -1,  1,  1, color='green',      linewidth=1.5, arrow_length_ratio=0.10)
ax.quiver(0, 0, 0,  1, -1, -1, color='orange',     linewidth=1.5, arrow_length_ratio=0.10)

from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Voxel Vector 1',
           markerfacecolor='blue', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Voxel Vector 2',
           markerfacecolor='green', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Voxel Vector 3',
           markerfacecolor='orange', markersize=8)
]

ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))

# -------------------------
# PLOT SETTINGS
# -------------------------
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)
ax.set_box_aspect([1, 1, 1])

# remove default axis labels since you use custom text labels
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_zlabel('')

# better viewing angle
ax.view_init(elev=22, azim=-20)

plt.tight_layout()
plt.show()