"""`python -m polony_app` : run the local server on 127.0.0.1:8000."""
from __future__ import annotations

import argparse
import os


def main() -> None:
    p = argparse.ArgumentParser(prog="polony_app", description="Polony counting app server")
    p.add_argument("--data-dir", default=os.environ.get("POLONY_DATA_DIR", "./data"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    import uvicorn

    from polony_app.api.main import create_app

    uvicorn.run(create_app(args.data_dir), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
