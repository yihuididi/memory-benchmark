"""Small command-line entry point."""

import argparse
import sys
from pathlib import Path

from memory_bench.config import load_config
from memory_bench.runner import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare memory backends using a shared agent.")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("run", help="Run a benchmark × memory experiment matrix")
    command.add_argument("--config", type=Path, required=True, help="Path to the experiment TOML")
    args = parser.parse_args()
    try:
        output = run(load_config(args.config))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", ()):
            print(note, file=sys.stderr)
        return 1
    print(f"Results: {output}")
    return 0
