"""Save each panel of a multi-panel matplotlib figure as a separate image, for LaTeX subfigures.

save_panels(fig, path) writes, next to `path` (the file of the whole figure):
  <stem>_a<ext>, <stem>_b<ext>, ...   one file per panel, in reading order (rows top to bottom, then columns);
                                      a panel contains its axes, its title, labels, legend and its colorbar(s);
  <stem>_legend<ext>                  the figure-level legend(s) (fig.legend), if any;
  <stem>_panels.json                  figure size and size of each panel image in inches, used to set the widths
                                      of the subfigures so that the fonts keep the size of the whole figure.
Axes switched off (ax.axis('off') or invisible) are skipped.
"""
import json
import os
import string

from matplotlib.transforms import Bbox


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


def save_panels(fig, path, pad=0.04, **savefig_kw):
    stem, ext = os.path.splitext(path)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_inch = fig.dpi_scale_trans.inverted()
    artists = list(fig.axes) + list(fig.legends)
    visible = {id(a): a.get_visible() for a in artists}
    info = {'figsize': [float(x) for x in fig.get_size_inches()], 'panels': []}

    def save_only(keep, fname):
        for a in artists:
            a.set_visible(id(a) in {id(k) for k in keep} and visible[id(a)])
        bbox = Bbox.union([k.get_tightbbox(renderer) for k in keep]).transformed(to_inch).padded(pad)
        fig.savefig(fname, bbox_inches=bbox, **savefig_kw)
        return bbox

    try:
        for letter, (ax, cbs) in zip(string.ascii_lowercase, _panels(fig)):
            ss = ax.get_subplotspec()
            row = ss.get_topmost_subplotspec().rowspan.start if ss is not None else None
            fname = f"{stem}_{letter}{ext}"
            bbox = save_only([ax] + cbs, fname)
            info['panels'].append({'file': os.path.basename(fname), 'letter': letter,
                                   'width': bbox.width, 'height': bbox.height,
                                   'row': row})
        if fig.legends:
            fname = f"{stem}_legend{ext}"
            bbox = save_only(list(fig.legends), fname)
            info['legend'] = {'file': os.path.basename(fname), 'width': bbox.width, 'height': bbox.height}
    finally:
        for a in artists:
            a.set_visible(visible[id(a)])
    with open(f"{stem}_panels.json", 'w') as fh:
        json.dump(info, fh, indent=1)
    return info
