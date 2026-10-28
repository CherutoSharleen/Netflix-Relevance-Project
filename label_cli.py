"""Expose the human-only judgment workflow without mixing it with retrieval tuning."""

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from film_data import DATA_PATH, load_records
from judgments import consistency, create_pool, label_pool, sample_relabel
from query_sets import SPLITS, freeze_queries


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split", help="Freeze human-authored queries before tuning")
    split.add_argument("--queries", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--heldout-fraction", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=42)
    pool = commands.add_parser("pool", help="Prepare ungraded baseline candidates")
    pool.add_argument("--output", type=Path, required=True)
    pool.add_argument("--partition", choices=SPLITS, default="dev")
    pool.add_argument("-k", type=int, default=10)
    pool.add_argument("--seed", type=int, default=42)
    pool.add_argument("--additional-pairs", type=Path, help="Optional ungraded query_id,show_id CSV")
    label = commands.add_parser("label", help="Enter or resume explicit human grades")
    label.add_argument("--pool", type=Path, required=True)
    sample = commands.add_parser("sample", help="Save a blind re-label subset for a later session")
    sample.add_argument("--pool", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--size", type=int, required=True)
    sample.add_argument("--seed", type=int, default=43)
    summary = commands.add_parser("consistency", help="Compare completed human passes")
    summary.add_argument("--pool", type=Path, required=True)
    for command in (split, pool, label, sample, summary):
        command.add_argument("--csv", type=Path, default=DATA_PATH)
    for command in (pool, label, sample, summary):
        command.add_argument("--split-file", type=Path, required=True)
    for command in (label, sample, summary):
        command.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "split":
            load_records(args.csv)
            freeze_queries(args.queries, args.output, args.csv, args.heldout_fraction, args.seed)
            print("Query split frozen. Use only dev queries for tuning; keep heldout sealed.")
        elif args.command == "pool":
            create_pool(args.split_file, args.csv, args.output, args.partition, args.k,
                        args.seed, args.additional_pairs)
            print("Ungraded candidate pool saved. No relevance labels were generated.")
        elif args.command == "label":
            label_pool(args.pool, args.split_file, args.csv, args.labels)
        elif args.command == "sample":
            sample_relabel(args.pool, args.split_file, args.csv, args.labels,
                           args.output, args.size, args.seed)
            print("Blind re-label sample saved. Return later and label this saved pool.")
        else:
            print(json.dumps(consistency(args.pool, args.split_file, args.csv, args.labels), indent=2))
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
