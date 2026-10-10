"""Import the libraries the app needs; exit code 1 (with a message) when one is missing."""
import sys

try:
    import bigfish  # noqa: F401
    import fastapi  # noqa: F401
    import openpyxl  # noqa: F401
    import psycopg  # noqa: F401
    import psycopg_pool  # noqa: F401
    import scipy  # noqa: F401
    import skimage  # noqa: F401
    import tifffile  # noqa: F401
    import uvicorn  # noqa: F401
except Exception as e:  # pragma: no cover
    print(f"import failed: {type(e).__name__}: {e}")
    sys.exit(1)
print(f"libraries import fine (big-fish {bigfish.__version__})")
