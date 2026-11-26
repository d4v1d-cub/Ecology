"""Save each panel of a multi-panel matplotlib figure as a separate image of fixed size, for LaTeX subfloats.

save_panels(fig, path) does the following:
  - removes the panel label at the start of every title ("(a) ...", "(b)", ...): the labels are added in LaTeX
    (\\subfloat), never in the images;
  - saves the whole figure, without panel labels, at `path` (with "." replaced by "p" in the file name);
  - saves every panel as <stem>_<name><ext>, with the names given in `names`, in reading order (rows top to bottom,
    then columns). The names describe the content of the panel through its parameters (for example
    "mu_0.370_sigma_0.000"), not its position in a document. Every "." in the stem and in the names is replaced
    by "p" (0.370 -> 0p370), so that the file names contain no dot except the one of the extension.
    Every panel image has the same size, PANEL_SIZE, and the axes box has the same position and size in all of
    them (a narrower box when the panel has a colorbar, which is drawn at a fixed place on the right);
  - saves the figure-level legend(s) (fig.legend), if any, as <stem>_legend<ext> (tight box);
  - writes <stem>_panels.json with the list of panels and their rows.
Axes switched off (ax.axis('off') or invisible) are skipped. The function prints a warning when a panel does not
fit in the fixed image (tick labels or axis labels cut).
"""
import json
import os
import re

from matplotlib.transforms import Bbox

PANEL_SIZE = (5.0, 4.0)                 # inches, every panel image
AXES_RECT = (1.2, 0.72, 3.6, 2.9)       # left, bottom, width, height of the axes box, inches
CBAR_WIDTH, CBAR_GAP, CBAR_ROOM = 0.15, 0.12, 0.6   # colorbar width, gap to the axes, room for its labels (inches)
_LABEL = re.compile(r'^\s*\([a-z]\)\s*')


def strip_panel_labels(fig):
    for ax in fig.axes:
        for loc in ('center', 'left', 'right'):
            t = ax.get_title(loc=loc)
            if t and _LABEL.match(t):
                ax.set_title(_LABEL.sub('', t), loc=loc)


def _panels(fig):
    """main axes in reading order, each with the list of its colorbar axes"""
    cbars = [ax for ax in fig.axes if hasattr(ax, '_colorbar')]
    main = [ax for ax in fig.axes if ax not in cbars and ax.get_visible() and ax.axison]

    def key(ax):
        ss = ax.get_subplotspec()
        if ss is not None:
            ss = ss.get_topmost_subplotspec()
            return (ss.rowspan.start, ss.colspan.start)
        p = ax.get_position()
        return (-round(p.y1, 2), round(p.x0, 2))
    main.sort(key=key)
    owned = {id(ax): [] for ax in main}
    for cb in cbars:
        parent = cb._colorbar.mappable.axes
        if id(parent) in owned:
            owned[id(parent)].append(cb)
    return [(ax, owned[id(ax)]) for ax in main]


def nodot(s):
    return str(s).replace('.', 'p')


def save_panels(fig, path, names, **savefig_kw):
    strip_panel_labels(fig)
    folder, base = os.path.split(path)
    base, ext = os.path.splitext(base)
    stem = os.path.join(folder, nodot(base))
    fig.savefig(stem + ext, **savefig_kw)
    panels = _panels(fig)
    if len(names) != len(panels):
        raise ValueError(f"save_panels: {len(panels)} panels but {len(names)} names for {base}")
    artists = list(fig.axes) + list(fig.legends)
    visible = {id(a): a.get_visible() for a in artists}
    size0 = fig.get_size_inches().copy()
    pos0 = {id(ax): (ax.get_position(original=True), ax.get_position(original=False), ax.get_axes_locator())
            for ax in fig.axes}
    W, H = PANEL_SIZE
    info = {'panel_size': list(PANEL_SIZE), 'panels': []}
    try:
        for name, (ax, cbs) in zip(names, panels):
            for a in artists:
                a.set_visible(a is ax or a in cbs)
            fig.set_size_inches(W, H)
            l, b, w, h = AXES_RECT
            if cbs:
                w -= CBAR_GAP + CBAR_WIDTH + CBAR_ROOM
            ax.set_axes_locator(None)
            ax.set_position([l / W, b / H, w / W, h / H], which='both')
            for k, cb in enumerate(cbs):
                cb.set_axes_locator(None)
                x = l + w + CBAR_GAP + k * (CBAR_WIDTH + CBAR_ROOM)
                cb.set_position([x / W, b / H, CBAR_WIDTH / W, h / H], which='both')
            fname = f"{stem}_{nodot(name)}{ext}"
            fig.savefig(fname, **savefig_kw)
            renderer = fig.canvas.get_renderer()
            bb = Bbox.union([a.get_tightbbox(renderer) for a in [ax] + cbs]).transformed(fig.dpi_scale_trans.inverted())
            if bb.x0 < -0.01 or bb.y0 < -0.01 or bb.x1 > W + 0.01 or bb.y1 > H + 0.01:
                print(f"save_panels: {os.path.basename(fname)} exceeds the image: "
                      f"x {bb.x0:.2f}..{bb.x1:.2f} (0..{W}), y {bb.y0:.2f}..{bb.y1:.2f} (0..{H})")
            ss = ax.get_subplotspec()
            row = ss.get_topmost_subplotspec().rowspan.start if ss is not None else None
            info['panels'].append({'file': os.path.basename(fname), 'name': name, 'row': row})
    finally:
        fig.set_size_inches(*size0)
        for a in artists:
            a.set_visible(visible[id(a)])
        for a in fig.axes:
            p_orig, p_act, loc = pos0[id(a)]
            a.set_position(p_orig, which='original')
            a.set_position(p_act, which='active')
            a.set_axes_locator(loc)
    if fig.legends:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for a in artists:
            a.set_visible(a in fig.legends and visible[id(a)])
        bbox = Bbox.union([lg.get_tightbbox(renderer) for lg in fig.legends]).transformed(
            fig.dpi_scale_trans.inverted()).padded(0.04)
        fname = f"{stem}_legend{ext}"
        fig.savefig(fname, bbox_inches=bbox, **savefig_kw)
        for a in artists:
            a.set_visible(visible[id(a)])
        info['legend'] = {'file': os.path.basename(fname), 'width': bbox.width, 'height': bbox.height}
    with open(f"{stem}_panels.json", 'w') as fh:
        json.dump(info, fh, indent=1)
    return info


# Colours for the number of initial conditions n_init, the same in every figure: matplotlib's default colour cycle
# (C0 blue for 256, C1 orange for 512, ..., C7 grey for 32768).
NINIT_ALL = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)


def ninit_color(n):
    k = NINIT_ALL.index(int(n))
    return f"C{k}"
