import matplotlib.pyplot as plt
import numpy as np

configs = [(500, 512), (500, 256), (250, 512), (250, 256)]
min_widths = [32, 64, 128, 256]
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

for ax, (n_steps, init_width) in zip(axes.flatten(), configs):
    steps = np.arange(n_steps)
    for w in min_widths:
        widths = np.maximum(w, (init_width * (1 - steps / n_steps)).astype(int))
        ax.plot(steps, widths, label=f"Min: {w}")
    ax.set(
        title=f"Init: {init_width}, Steps: {n_steps}",
        xlabel="Step",
        ylabel="Search Width",
    )
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.savefig("search_width_comparison.png")
plt.show()
print("Plot saved at: search_width_comparison.png")
