"""Create the first admin user for server edition. Run on the customer machine."""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from private_analyst.auth.users import UserStore
from private_analyst.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a local admin user")
    parser.add_argument("--org", required=True)
    parser.add_argument("--username", required=True)
    args = parser.parse_args()
    password = getpass.getpass("Password (min 12 characters): ")
    cfg = load_config()
    store = UserStore(cfg.resolved_data_dir() / "app.db")
    user = store.create_user(args.org, args.username, password, "admin")
    print(f"Created admin {user.username} in org {user.org_id}")
    print("The password is stored as a bcrypt hash only.")


if __name__ == "__main__":
    os.environ.setdefault("PRIVATE_ANALYST_EDITION", "server")
    main()
