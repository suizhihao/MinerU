import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def _default_entry() -> Optional[str]:
    return os.getenv("MINERU_LIFECYCLE_REPAIR_ENTRY")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run an external training repair script (user-supplied entry point). "
            "Set MINERU_LIFECYCLE_REPAIR_ENTRY or pass --entry."
        )
    )
    parser.add_argument(
        "--entry",
        default=None,
        help="Path to Python training repair script. Overrides MINERU_LIFECYCLE_REPAIR_ENTRY.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to write run metadata JSON.",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Working directory for subprocess. Defaults to entry parent.",
    )
    parser.add_argument(
        "pass_through",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the entry script. Prefix with '--'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    entry_str = args.entry or _default_entry()
    if not entry_str:
        raise SystemExit(
            "Missing repair entry: set MINERU_LIFECYCLE_REPAIR_ENTRY or pass --entry /path/to/script.py"
        )
    entry = Path(entry_str).expanduser().resolve()
    if not entry.exists():
        raise FileNotFoundError(f"Repair entry not found: {entry}")
    cwd = Path(args.cwd).expanduser().resolve() if args.cwd else entry.parent

    pass_through = args.pass_through
    if pass_through and pass_through[0] == "--":
        pass_through = pass_through[1:]

    cmd = [sys.executable, str(entry), *pass_through]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{entry.parent}:{env.get('PYTHONPATH', '')}".rstrip(":")
    started_at = datetime.utcnow().isoformat() + "Z"
    completed = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    ended_at = datetime.utcnow().isoformat() + "Z"

    metadata = {
        "tool": "train_repair",
        "entry": str(entry),
        "cwd": str(cwd),
        "command": cmd,
        "return_code": completed.returncode,
        "started_at": started_at,
        "ended_at": ended_at,
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
