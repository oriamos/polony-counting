# Attribution for the historical-TIFF import

`polony_app.legacy_import` reads the laboratory's historical ImageJ-annotated TIFFs (embedded `ROI` and `Overlays`
blobs, counter positions, counting-grid lines, the `<batch>-<n>-<A/B>.tif` naming and the 158 / 316 px grid origin).
Those file conventions were learned from the lab's files and from the earlier counting scripts that produced them
(https://github.com/kirill-push/polony-counting, MIT). **No code was copied**; every parser here is new code with its
own tests, and it differs on purpose: per-blob error isolation, an explicit canonical coordinate frame, no class
assumptions, grid lines kept as floats with provenance, deterministic group-based splits instead of a sequential 90/10
split.

If code is ever copied from another project, record the file, commit and modifications here.
