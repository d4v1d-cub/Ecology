"""Paths. Override with environment variables, e.g.
   VECTORS_DIR=/path/to/data_vectors HIST_DIR=/path/to/Pq_histograms python analyze_vectors.py
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
# Unzipped data_vectors.zip: IBMF_T0_..._mu_<mu>_sigma_0.000_N_<N>_..._seedinit_<k>.txt (column 2 = av(n))
VECTORS_DIR = os.environ.get("VECTORS_DIR", os.path.join(HERE, "..", "data_vectors"))
# Pairwise-distance histograms (Windows: D:\Research\Ecology\Results\IBMF\Counting_fixed_points\Pq_histograms)
HIST_DIR = os.environ.get("HIST_DIR", os.path.join(HERE, "..", "Pq_histograms"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(HERE, "..", "results"))
FIG_DIR = os.environ.get("FIG_DIR", os.path.join(HERE, ".."))
os.makedirs(RESULTS_DIR, exist_ok=True)

# Colours used in all figures (validated categorical palette)
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#8a8a85"
def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED})
