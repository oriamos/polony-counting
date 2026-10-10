"""Store the optional shared-database URL in <POLONY_HOME>/config.json (used by the installers).

The URL is read from STDIN only (never from arguments or the environment, so it is not visible in process
lists) and written with polony_app.config.save_config_file (atomic, mode 0600; on Windows the ACL is then
restricted to the current user). Other keys of config.json are kept; machine_id is never created or copied here.

    set_database.py status          exit 0 = a database_url is already configured, 1 = none
    set_database.py set [--replace] URL on stdin; an existing URL is kept unless --replace
    set_database.py clear
"""
import subprocess
import sys

from polony_app.config import config_path, load_config_file, mask_url, save_config_file


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "status"
    cfg = load_config_file()
    have = isinstance(cfg.get("database_url"), str) and bool(cfg["database_url"].strip())
    if cmd == "status":
        if have:
            print(f"configured: {mask_url(cfg['database_url'])}")
        return 0 if have else 1
    if cmd == "clear":
        cfg.pop("database_url", None)
        save_config_file(cfg)
        print("shared database removed")
        return 0
    if cmd == "set":
        url = sys.stdin.readline().strip()
        if not url.startswith(("postgresql://", "postgres://")) or "@" not in url:
            print("That is not a PostgreSQL connection string (expected postgresql://user:password@host/db); nothing saved.")
            return 2
        if have and "--replace" not in argv:
            print("A shared database is already configured; kept it.")
            return 3
        cfg["database_url"] = url
        save_config_file(cfg)
        if sys.platform == "win32":
            import os

            user = os.environ.get("USERNAME")
            if user:
                subprocess.run(["icacls", str(config_path()), "/inheritance:r", "/grant:r", f"{user}:(R,W)"],
                               capture_output=True, timeout=30)
        print(f"shared database saved: {mask_url(url)}")
        return 0
    print(f"unknown command {cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
