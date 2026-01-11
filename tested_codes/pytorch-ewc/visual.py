import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from torch.cuda import FloatTensor as CUDATensor
import os

# Directory to save plots
_PLOTS_DIR = './plots'
os.makedirs(_PLOTS_DIR, exist_ok=True)

# Store plot data for updating
_PLOT_DATA = {}


def visualize_image(vis, tensor, name, label=None, w=250, h=250,
                    update_window_without_label=False):
    """Visualize a single image - kept for compatibility but not used with PNG saving."""
    pass


def visualize_images(vis, tensor, name, label=None, w=250, h=250,
                     update_window_without_label=False):
    """Visualize multiple images - kept for compatibility but not used with PNG saving."""
    pass


def visualize_kernel(vis, kernel, name, label=None, w=250, h=250,
                     update_window_without_label=False, compress_tensor=False):
    """Visualize kernel - kept for compatibility but not used with PNG saving."""
    pass


def visualize_scalar(vis, scalar, name, iteration):
    """Visualize a single scalar value."""
    visualize_scalars(
        vis,
        [scalar] if isinstance(scalar, float) or len(scalar) == 1 else scalar,
        [name], name, iteration
    )


def visualize_scalars(vis, scalars, names, title, iteration):
    """Visualize multiple scalars and save as PNG file."""
    assert len(scalars) == len(names)
    # Convert scalar tensors to numpy arrays.
    scalars, names = list(scalars), list(names)
    scalars = [s.cpu() if isinstance(s, CUDATensor) else s for s in scalars]
    scalars = [s.detach().numpy() if hasattr(s, 'numpy') else
               np.array([s]) for s in scalars]
    scalars = [float(s) if np.isscalar(s) or s.size == 1 else s[0] for s in scalars]
    
    # Initialize plot data if needed
    if title not in _PLOT_DATA:
        _PLOT_DATA[title] = {
            'iterations': [],
            'values': {name: [] for name in names}
        }
    
    # Append new data
    _PLOT_DATA[title]['iterations'].append(iteration)
    for name, value in zip(names, scalars):
        _PLOT_DATA[title]['values'][name].append(value)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot each line
    for name in names:
        plt.plot(
            _PLOT_DATA[title]['iterations'],
            _PLOT_DATA[title]['values'][name],
            label=name,
            linewidth=2
        )
        # Fill area under the curve (only for first curve if multiple)
        if len(names) == 1:
            plt.fill_between(
                _PLOT_DATA[title]['iterations'],
                _PLOT_DATA[title]['values'][name],
                alpha=0.3
            )
    
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel(title, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    filename = title.replace(' ', '_').replace('(', '').replace(')', '').lower()
    filepath = os.path.join(_PLOTS_DIR, f'{filename}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'  => Saved plot: {filepath}')
