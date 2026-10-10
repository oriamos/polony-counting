"""Import of historical (ImageJ-annotated) polony data into a canonical, versioned dataset.

See docs/legacy-repo-audit.md (findings) and docs/legacy-import.md (usage).
Entry point: ``python -m polony_app.legacy_import --help``.
"""
from .importer import ImportConfig, ImportReport, build_dataset, discover  # noqa: F401
from .schema import SCHEMA_NAME, SCHEMA_VERSION, Dataset, read_dataset, write_dataset  # noqa: F401
