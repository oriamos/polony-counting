"""Dump TIFF structure, ImageJ metadata and embedded ROI/Overlays of real slides.
Usage: python -I inspect_meta.py <tif> [<tif> ...]"""
import sys
import numpy as np
import tifffile
import roifile

for p in sys.argv[1:]:
    print("=" * 100); print(p)
    with tifffile.TiffFile(p) as t:
        print("is_imagej", t.is_imagej, "pages", len(t.pages), "series", [(s.shape, str(s.dtype), s.axes) for s in t.series])
        for i, pg in enumerate(t.pages):
            tags = {v.name: (v.value if len(str(v.value)) < 300 else str(v.value)[:300] + '...') for v in pg.tags.values()
                    if v.name not in ("StripOffsets", "StripByteCounts", "IJMetadata", "IJMetadataByteCounts")}
            print(f" page{i}: shape {pg.shape} dtype {pg.dtype} photometric {pg.photometric}")
            for k, v in tags.items():
                print(f"    {k}: {v!r}")
        md = t.imagej_metadata or {}
        for k, v in md.items():
            if k in ("ROI", "Overlays"):
                continue
            s = repr(v)
            print(f"  IJ {k}: {s[:1500]}")
        blobs = []
        if md.get("ROI"):
            blobs.append(("ROI", md["ROI"]))
        ov = md.get("Overlays")
        if ov is not None:
            for b in (ov if isinstance(ov, (list, tuple)) else [ov]):
                blobs.append(("Overlays", b))
        for src, b in blobs:
            r = roifile.ImagejRoi.frombytes(b)
            print(f"  {src}: type={r.roitype.name} name={r.name!r} subtype={r.subtype} n={r.n_coordinates} "
                  f"bbox=({r.left},{r.top},{r.right},{r.bottom}) pos c/z/t={r.c_position}/{r.z_position}/{r.t_position} "
                  f"position={r.position} stroke={r.stroke_color} fill={r.fill_color} opts={r.options} props={r.properties!r}")
            if r.counters is not None:
                print("     counters:", np.unique(np.asarray(r.counters), return_counts=True))
            if r.counter_positions is not None:
                print("     counter_positions:", np.unique(np.asarray(r.counter_positions), return_counts=True))
            if r.composite:
                print("     composite n subrois", len(r.subpaths() if hasattr(r,'subpaths') else []))
            if r.roitype.name != "POINT":
                c = r.coordinates()
                if isinstance(c, list):
                    print("     subpaths", len(c), [np.asarray(x).shape for x in c[:3]])
                else:
                    print("     coords", np.asarray(c)[:8])
