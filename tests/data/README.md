# Test fixtures

Small binary files the suite reads rather than builds at run time, because constructing
them depends on the environment in ways the behaviour under test does not.

## `ecf_like.h5` (12.6 KB)

An HDF5 file whose `CD/events` dataset declares the **ECF filter (36559)** as mandatory —
the pipeline a Metavision "compress on save" recording carries. Reading it without that
codec registered fails exactly as a real one does, which is what
`tests/test_hdf5.py::test_ecf_error_mentions_optin_install` asserts against.

It is committed rather than generated per-run because only HDF5 builds that defer the
filter check to write time can *create* a dataset demanding an unregistered mandatory
filter; other builds refuse outright with

    ValueError: Unable to synchronously create dataset (required filter 36559 is not registered)

which fails the test on the environment instead of on the behaviour under test.

Regenerate (on an HDF5 build that permits the creation):

```python
import h5py, numpy as np
from gottlux.io import hdf5 as h5io

n = 100
ev = np.zeros(n, h5io.EVENT_DTYPE)
ev["t"] = np.arange(n)
with h5py.File("ecf_like.h5", "w") as f:
    g = f.create_group("CD")
    space = h5py.h5s.create_simple((n,), (n,))
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_chunk((n,))
    dcpl.set_filter(36559, h5py.h5z.FLAG_MANDATORY)
    tid = h5py.h5t.py_create(h5io.EVENT_DTYPE, logical=True)
    dsid = h5py.h5d.create(g.id, b"events", tid, space, dcpl=dcpl)
    h5py.Dataset(dsid).id.write_direct_chunk((0,), ev.tobytes(), filter_mask=0)
    f.attrs["geometry"] = "320x320"
```
