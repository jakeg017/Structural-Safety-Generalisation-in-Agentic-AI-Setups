"""
view_logs.py

Start the inspect_ai log viewer.

Opens a web interface at http://127.0.0.1:7575 showing all eval logs in logs/.

Usage:
    python view_logs.py [--log-dir DIR] [--port PORT]

Alternative (CLI):
    inspect view --log-dir logs/
"""

import argparse

from inspect_ai import view


def main():
    parser = argparse.ArgumentParser(description="Start the inspect_ai log viewer.")
    parser.add_argument("--log-dir", default="logs/", help="Directory containing eval logs")
    parser.add_argument("--port", type=int, default=7575, help="Server port")
    args = parser.parse_args()

    print(f"Starting inspect view at http://127.0.0.1:{args.port}")
    print(f"Log directory: {args.log_dir}")
    print("Press Ctrl+C to stop.\n")

    view(log_dir=args.log_dir, port=args.port)


if __name__ == "__main__":
    main()
