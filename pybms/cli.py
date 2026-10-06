"""Command-line discovery and explicit syntax-file repair."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    import pybms as pb

    parser = argparse.ArgumentParser(
        prog="pybms", description="Discover Pybms functions or review syntax repairs."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    catalog = commands.add_parser("functions", help="List documented workflows")
    catalog.add_argument("--category")
    catalog.add_argument("--search")
    help_parser = commands.add_parser("help", help="Explain a function")
    help_parser.add_argument("name")
    fix = commands.add_parser("fix", help="Show repairs; write only to an explicit new output file")
    fix.add_argument("path", type=Path)
    fix.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "functions":
        print(pb.list_functions(args.category, search=args.search).to_string(index=False))
    elif args.command == "help":
        try:
            info = pb.function_help(args.name)
        except pb.ConfigurationError as exc:
            parser.error(str(exc))
        print(info["signature"] + "\n\n" + info["description"] + "\n\nExample:\n" + info["example"])
    else:
        result = pb.fix_syntax(args.path.read_text(encoding="utf-8"))
        if not result.valid:
            parser.exit(1, f"Unresolved syntax error: {result.error}\n")
        if args.output:
            if args.output.resolve() == args.path.resolve() or args.output.exists():
                parser.error(
                    "Choose a new output filename; input and existing files are preserved."
                )
            args.output.write_text(result.source, encoding="utf-8")
            print(f"Wrote {args.output}")
        else:
            print(result.source)
        for change in result.changes:
            print("Repair: " + change)
    return 0
