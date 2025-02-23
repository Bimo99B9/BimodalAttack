import os
import matplotlib.pyplot as plt
import numpy as np

# Define different configurations
configurations = [
    (500, 512),
    (500, 256),
    (250, 512),
    (250, 256),
]

# Define minimum search widths
min_search_widths = [32, 64, 128, 256]


# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()

# Generate plots for each configuration
for ax, (num_steps, initial_search_width) in zip(axes, configurations):
    steps = np.arange(num_steps)
    for min_w in min_search_widths:
        widths = [max(min_w, int(initial_search_width * (1 - i / num_steps))) for i in steps]
        ax.plot(steps, widths, label=f"Min: {min_w}")
    
    ax.set_title(f"Init: {initial_search_width}, Steps: {num_steps}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Search Width")
    ax.legend()
    ax.grid(True)

# Adjust layout and save figure
plt.tight_layout()
plot_path = "search_width_comparison.png"
plt.savefig(plot_path)
plt.show()

print(f"Plot saved at: {plot_path}")