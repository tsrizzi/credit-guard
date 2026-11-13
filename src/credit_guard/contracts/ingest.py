"""Porta de ingestao: valida um lote pelo contrato e BLOQUEIA (exit 1) se violar.

Uso:
    python -m credit_guard.contracts.ingest --input lote.parquet --report-dir reports/validation --name lote
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from credit_guard.contracts.report import write_validation_report
from credit_guard.contracts.validation import validate_batch
from credit_guard.logging_utils import get_logger
from credit_guard.observability.pipeline import track_stage

log = get_logger("ingest")


def read_batch(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Valida um lote pelo contrato de dados.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--include-protected", action="store_true")
    parser.add_argument(
        "--expect-reject",
        action="store_true",
        help="modo demonstracao: sucesso (0) quando o lote E bloqueado",
    )
    args = parser.parse_args(argv)

    with track_stage(f"ingest_{args.name}") as ctx:
        result = validate_batch(read_batch(args.input), include_protected=args.include_protected)
        ctx.add_rows("accepted" if result.passed else "blocked", result.n_rows)
    json_path, html_path = write_validation_report(result, args.report_dir, args.name)
    extra = {
        "batch": args.name,
        "rows": result.n_rows,
        "invalid_rows": result.n_invalid_rows,
        "violations_by_check": result.violations_by_check,
        "report": str(html_path),
    }
    if result.passed:
        log.info("ingestion_accepted", extra=extra)
    else:
        log.warning("ingestion_blocked", extra=extra)

    blocked = not result.passed
    if args.expect_reject:
        return 0 if blocked else 1
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
