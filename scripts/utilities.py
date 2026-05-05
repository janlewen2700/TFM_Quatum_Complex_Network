import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def set_article_style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        
        # Standardize sizes for 2-column layout
        "font.size": 10,           # Matches RevTeX 10pt
        "axes.labelsize": 10,      # Slightly larger for readability
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,

        # Thicker lines for visibility (This is better than bolding text)
        "axes.linewidth": 1.2,
        "lines.linewidth": 1.5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,         # Recommended for PRX/PRA
        "ytick.right": True,       # Recommended for PRX/PRA
        
        "axes.spines.top": True,    # I suggest keeping these for "box" style
        "axes.spines.right": True,  # typical in Physics journals
        
        "figure.figsize": [3.375, 2.5], # Standard column width in inches
        # Fixed margins for the figure
        "figure.subplot.left": 0.15,   # Room for the Y-axis label
        "figure.subplot.bottom": 0.15, # Room for the X-axis label
        "figure.subplot.right": 0.98,  # Small margin on the right
        "figure.subplot.top": 0.94,     # Small margin on the top
        
        # Disable autolayout if you use fixed margins
        "figure.autolayout": False,
        
        # Still use tight bbox for saving just in case
        "savefig.bbox": "tight",
        
        "savefig.dpi": 600,
        "savefig.format": "pdf",    # Always save as PDF for LaTeX
    })
    
