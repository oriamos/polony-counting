Research samples uploaded by the user: eight original TIFF images (stored in lossless ZIP archives) and two reference CSVs.

Extract each ZIP in this directory to recover the original TIFF filename and bytes. For example:

```python
from pathlib import Path
import zipfile
root = Path('resources/dataset_files/research/2026-09-29')
for archive in root.glob('*.tif.zip'):
    with zipfile.ZipFile(archive) as z:
        z.extractall(root)
```

Channel identity and the meaning of T7A/T7B must be confirmed with the researcher. Blank CSV fields are not measured zeros. The supplied CSV sample volume is 2.5 microliters.
