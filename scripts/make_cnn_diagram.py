import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from PIL import Image
import numpy as np
from pathlib import Path


def add_arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start, end,
            arrowstyle='simple',
            mutation_scale=20,
            color='black',
            linewidth=1.5
        )
    )

def draw_conv_block(ax, x, y, w=0.9, h=1.2, depth=3, offset=0.18):
    for i in range(depth):
        rect = Rectangle(
            (x + i * offset, y + i * offset),
            w, h,
            facecolor='#8ecae6',
            edgecolor='#1d4e89',
            linewidth=1.5,
            alpha=0.8
        )
        ax.add_patch(rect)

def draw_feature_maps(ax, x, y, n=6, size=0.55, offset=0.13):
    colors = ['#cdb4db', '#bde0fe', '#ffc8dd', '#bdb2ff']
    for i in range(n):
        rect = Rectangle(
            (x + (i % 3) * offset, y + i * offset),
            size, size,
            facecolor=colors[i % len(colors)],
            edgecolor='#6a4c93',
            linewidth=1.0,
            alpha=0.75
        )
        ax.add_patch(rect)

def draw_fc_layer(ax, x, y, input_nodes=6, hidden_nodes=5, output_nodes=2):
    # Input column
    in_y = np.linspace(y + 1.6, y + 0.2, input_nodes)
    hidden_y = np.linspace(y + 1.5, y + 0.3, hidden_nodes)
    out_y = np.linspace(y + 1.2, y + 0.6, output_nodes)

    in_x = x
    hidden_x = x + 0.9
    out_x = x + 1.8


    for iy in in_y:
        for hy in hidden_y:
            ax.plot([in_x, hidden_x], [iy, hy], color='gray', linewidth=0.8, alpha=0.6)

  
    for hy in hidden_y:
        for oy in out_y:
            ax.plot([hidden_x, out_x], [hy, oy], color='gray', linewidth=0.8, alpha=0.6)


    for iy in in_y:
        ax.add_patch(Circle((in_x, iy), 0.06, color='#2a9d8f'))
    for hy in hidden_y:
        ax.add_patch(Circle((hidden_x, hy), 0.06, color='#2a9d8f'))
    for oy in out_y:
        ax.add_patch(Circle((out_x, oy), 0.07, color='#e76f51'))

def draw_prediction_box(ax, x, y):
    box = Rectangle((x, y), 1.5, 0.9, facecolor='#d9f0d3', edgecolor='#4f772d', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + 0.75, y + 0.45, "Prediction", ha='center', va='center', fontsize=16, fontweight='bold')

def add_label(ax, x, y, text, size=16):
    ax.text(x, y, text, ha='center', va='top', fontsize=size, fontweight='bold')

# -----------------------------
# Main figure
# -----------------------------
fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 15)
ax.set_ylim(0, 6)
ax.axis('off')

# Load leaf image
img_path = Path("data/downloads/plantvillage/Tomato_healthy/e9392a5e-2fac-48c5-a23d-df3aa3f61048___GH_HL Leaf 217.JPG")
if img_path.exists():
    img = Image.open(img_path).convert("RGB")
    img = img.resize((220, 220))
    img_arr = np.array(img)
    ax.imshow(img_arr, extent=(0.4, 2.0, 2.0, 4.0))
else:
    # fallback if image missing
    ax.add_patch(Rectangle((0.4, 2.0), 1.6, 2.0, facecolor='lightgray', edgecolor='black'))
    ax.text(1.2, 3.0, "Leaf Image", ha='center', va='center', fontsize=14)

add_label(ax, 1.2, 1.65, "Leaf Image")

# Convolution layers
draw_conv_block(ax, 3.0, 2.15)
add_label(ax, 3.7, 1.65, "Convolution\nLayers")

# Feature maps
draw_feature_maps(ax, 5.7, 2.1)
add_label(ax, 6.4, 1.65, "Feature\nMaps")

# Fully connected layer
draw_fc_layer(ax, 8.8, 2.0)
add_label(ax, 9.7, 1.65, "Fully Connected\nLayer")

# Prediction box
draw_prediction_box(ax, 12.6, 2.55)

# Arrows
add_arrow(ax, (2.1, 3.0), (2.9, 3.0))
add_arrow(ax, (4.3, 3.0), (5.5, 3.0))
add_arrow(ax, (7.0, 3.0), (8.5, 3.0))
add_arrow(ax, (10.9, 3.0), (12.4, 3.0))

# Caption
ax.text(
    7.5, 0.5,
    "Simplified architecture of the transfer-learning CNN used for plant stress classification.",
    ha='center', va='center', fontsize=18
)

# Save
out_path = "outputs/cnn_architecture_diagram.png"
plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved diagram to {out_path}")