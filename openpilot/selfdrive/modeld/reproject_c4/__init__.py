"""Reproject the comma 3X cameras into comma 4 camera geometry, on the GPU, in front of an untouched comma 4 model.

Every comma 4 output pixel is traced through the comma 4 lens to a ray, rotated onto the 3X device axis and looked up in
the 3X wide (fisheye) or, for the narrow output, composited from the 3X narrow inset and the 3X wide surround. None of it
depends on the calibration, so the lookup tables are constants built once per process and the per-frame work is one
gather per output byte (two plus a blend for the composite) - the same cost as the model's own crop warp.

Outputs are NV12 buffers in the comma 4 camerad layout so the model pkl (keyed by its camera size) sees exactly what a
comma 4 would have given it. Lens numbers come from the bench + multi-device work in ~/tizi-to-mici.

geometry: lenses, rotations, sample coordinates. tables: the gather tables. rotation: the fitted rotation across boots.
fit: the rotation fit from frame pairs. meter: the seam exposure match. cameras: camerad's frame pairs. debug: the outlines.
kernel (the GPU stage, tinygrad) is imported by the stage alone."""
from .geometry import *  # noqa: F403
from .meter import *  # noqa: F403
from .tables import *  # noqa: F403
from .rotation import *  # noqa: F403
from .fit import *  # noqa: F403
from .cameras import *  # noqa: F403
from .debug import *  # noqa: F403
