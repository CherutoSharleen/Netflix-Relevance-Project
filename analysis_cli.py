"""Expose development ablations and human-reviewed failure reports with held-out guards."""

import argparse
import csv
from pathlib import Path
from typing import Sequence

from analysis import (
    evaluate, freeze_configuration, load_evaluation, load_experiments, load_frozen,
    prepare_pool, save_evaluation,
)
from failure_review import review_files, summary_files, write_bundle
from film_data import DATA_PATH, load_records
from query_sets import fingerprint


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Pool predeclared runs before human labelling")
    prepare.add_argument("--additional-pairs", type=Path)
    run = commands.add_parser("evaluate", help="Measure runs only with complete human grades")
    run.add_argument("--pool", type=Path, required=True)
    run.add_argument("--labels", type=Path, required=True)
    run.add_argument("--relevant-grade", type=int, choices=(1, 2, 3))
    run.add_argument("--failure-threshold", type=float)
    for command in (prepare, run):
        command.add_argument("--csv", type=Path, default=DATA_PATH)
        command.add_argument("--split-file", type=Path, required=True)
        command.add_argument("--partition", choices=("dev", "heldout"), default="dev")
        command.add_argument("-k", type=int)
        choice = command.add_mutually_exclusive_group(required=True)
        choice.add_argument("--experiments", type=Path)
        choice.add_argument("--frozen-config", type=Path)
    freeze = commands.add_parser("freeze", help="Record a human-selected development configuration")
    freeze.add_argument("--report", type=Path, required=True)
    freeze.add_argument("--experiment", required=True)
    review = commands.add_parser("review", help="Export below-threshold evidence and blank assignments")
    review.add_argument("--report", type=Path, required=True)
    review.add_argument("--csv", type=Path, default=DATA_PATH)
    summary = commands.add_parser("summarize", help="Count human-assigned causes and export a chart")
    summary.add_argument("--report", type=Path, required=True)
    summary.add_argument("--assignments", type=Path, required=True)
    for command in (prepare, run, freeze, review, summary):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            freeze_configuration(args.report, args.experiment, args.output)
        elif args.command in {"review", "summarize"}:
            report = load_evaluation(args.report)
            if args.command == "review":
                if fingerprint(args.csv) != report.corpus_sha256:
                    raise ValueError("Review corpus does not match the evaluated corpus")
                records, _ = load_records(args.csv)
                files = review_files(report, records)
            else:
                files = summary_files(report, args.assignments)
            write_bundle(args.output, files)
        else:
            if args.partition == "heldout":
                if args.frozen_config is None:
                    raise ValueError("Held-out work requires --frozen-config, not --experiments")
                settings = load_frozen(args.frozen_config, args.split_file, args.csv)
                experiments = [settings.experiment]
                k = settings.k if args.k is None else args.k
            else:
                if args.experiments is None:
                    raise ValueError("Development work requires --experiments")
                settings = None
                experiments = load_experiments(args.experiments)
                k = 10 if args.k is None else args.k
            if args.command == "prepare":
                prepare_pool(args.split_file, args.csv, experiments, args.partition,
                             args.output, k, args.additional_pairs, args.frozen_config)
            else:
                grade = args.relevant_grade
                threshold = args.failure_threshold
                if settings is not None:
                    grade = settings.relevant_grade if grade is None else grade
                    threshold = settings.failure_threshold if threshold is None else threshold
                if grade is None or threshold is None:
                    raise ValueError("Choose --relevant-grade and --failure-threshold explicitly")
                report = evaluate(
                    args.pool, args.split_file, args.csv, args.labels, experiments,
                    args.partition, k, grade, threshold, args.frozen_config,
                )
                save_evaluation(args.output, report)
        print(f"Saved {args.output}. No labels or failure causes were inferred.")
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
