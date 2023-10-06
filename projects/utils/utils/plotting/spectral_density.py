from typing import Optional

import numpy as np
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Bright6 as palette
from bokeh.plotting import figure

from utils.plotting.utils import save


def plot_psds(
    freqs: np.ndarray,
    pred: np.ndarray,
    strain: np.ndarray,
    clean: np.ndarray,
    asd: bool = True,
    fname: Optional[str] = None,
):
    source = ColumnDataSource(
        dict(
            x=freqs, raw=strain, pred=pred, cleaned=clean, ratio=clean / strain
        )
    )
    tooltips = [
        ("Frequency", "@x Hz"),
        ("Raw strain", "@raw"),
        ("Noise prediction", "@pred"),
        ("Cleaned", "@cleaned"),
        ("Ratio", "@ratio"),
    ]

    kwargs = dict(
        height=250,
        width=700,
        x_axis_label=r"$$\text{Frequency [Hz]}$$",
        y_axis_type="log",
        tools="box_zoom,save,reset",
    )

    y_axis_label = r"$$\text{{{}[Hz}}^{{-{}}}\text{{]}}$$".format(
        "ASD" if asd else "PSD", r"\frac{1}{2}" if asd else 1
    )
    p1 = figure(y_axis_label=y_axis_label, **kwargs)
    p1.xaxis.major_tick_line_color = None
    p1.xaxis.minor_tick_line_color = None
    p1.xaxis.major_label_text_font_size = "1pt"
    p1.xaxis.major_label_text_color = None

    def _plot_line(y, idx, label):
        return p1.line(
            "x",
            y,
            line_color=palette[idx],
            line_width=1.5,
            line_alpha=0.8,
            legend_label=label,
            source=source,
        )

    _plot_line("raw", 0, "Raw strain")
    _plot_line("pred", 1, "Noise prediction")
    r = _plot_line("cleaned", 2, "Cleaned")
    hover = HoverTool(renderers=[r], tooltips=tooltips, mode="vline")
    p1.add_tools(hover)

    p2 = figure(
        x_range=p1.x_range,
        y_axis_label=r"$$\text{Ratio (Cleaned / Raw)}$$",
        **kwargs
    )
    r = p2.line(
        "x",
        "ratio",
        line_color=palette[3],
        line_width=1.5,
        line_alpha=0.8,
        legend_label="Ratio",
        source=source,
    )
    p2.legend.location = "bottom_right"

    grid = gridplot([p1, p2], ncols=1, toolbar_location="right")
    if fname is not None:
        save(grid, fname, title="DeepClean PSDs")
    return grid
