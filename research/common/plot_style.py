"""Minimal shared figure style (rule book: PNG, at least 150 dpi, readable on a phone)."""
import matplotlib

DPI = 150


def apply_style():
    matplotlib.rcParams.update({"figure.dpi": 100, "savefig.dpi": DPI, "font.size": 12,
                                "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 11,
                                "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight"})
