"""
gottlux.studio — short, shareable animations of event-camera recordings.

A *shot* (:class:`Shot`) is a plain-data recipe: a span of a recording, a crop, a *look*
(one of the visualization methods in :mod:`gottlux.studio.looks`) with its parameters, a
playback speed and an output format. :class:`ShotRenderer` turns it into frames and
:func:`export_shot` writes a GIF / MP4 / WebM / PNG sequence plus a ``.shot.json`` recipe.

The same shot drives the desktop Studio tab, the ``gottlux-studio`` command and the web
Studio (``web/studio.html``). Method notes: ``docs/STUDIO.md``.

    >>> import gottlux
    >>> from gottlux.studio import Shot, export_shot
    >>> rec = gottlux.load("examples/data/Humming_Bird_Fight_merged_2.raw")
    >>> export_shot(rec, Shot(look="strobe", t0=1.0, t1=1.45, crop=(0, 110, 68, 178)),
    ...             "wingbeat.gif")                                   # doctest: +SKIP
"""
from gottlux.studio.export import FORMATS, default_folder, export_shot
from gottlux.studio.looks import COLORMAPS, Events, Look, get_look, list_looks, register
from gottlux.studio.render import ShotRenderer, render_preview
from gottlux.studio.shot import PALETTES, PRESETS, Shot

__all__ = ["COLORMAPS", "Events", "FORMATS", "Look", "PALETTES", "PRESETS", "Shot",
           "ShotRenderer", "default_folder", "export_shot", "get_look", "list_looks",
           "register", "render_preview"]
