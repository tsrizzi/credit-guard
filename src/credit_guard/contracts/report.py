"""Relatorio de validacao do contrato em JSON (maquina) e HTML (pessoas)."""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path

from credit_guard.contracts.validation import ValidationResult


def write_validation_report(
    result: ValidationResult, out_dir: Path, name: str
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"batch": name, "generated_at": datetime.now(UTC).isoformat(), **result.to_dict()}
    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    verdict = payload["verdict"]
    color = "#1a7f37" if result.passed else "#cf222e"
    by_check = (
        "".join(
            f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>"
            for k, v in payload["violations_by_check"].items()
        )
        or "<tr><td colspan=2>nenhuma violacao</td></tr>"
    )
    cases = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(c.get(k)))}</td>"
            for k in ("row_index", "column", "check", "failure_case")
        )
        + "</tr>"
        for c in payload["failure_cases"]
    )
    html_path = out_dir / f"{name}.html"
    html_path.write_text(
        f"""<!doctype html><html lang="pt-br"><head><meta charset="utf-8">
<title>Contrato de dados - {html.escape(name)}</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;max-width:1100px}}
table{{border-collapse:collapse;margin:1rem 0}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left}}
.verdict{{color:#fff;background:{color};padding:.4rem 1rem;border-radius:6px;display:inline-block}}</style>
</head><body>
<h1>Contrato de dados: <code>{html.escape(name)}</code></h1>
<p class="verdict"><strong>{verdict}</strong></p>
<p>Linhas: {result.n_rows} &middot; Linhas invalidas: {result.n_invalid_rows} &middot; Gerado em {payload["generated_at"]}</p>
<h2>Violacoes por regra</h2><table><tr><th>Regra</th><th>Ocorrencias</th></tr>{by_check}</table>
<h2>Primeiros casos</h2><table><tr><th>linha</th><th>coluna</th><th>regra</th><th>valor</th></tr>{cases}</table>
</body></html>""",
        encoding="utf-8",
    )
    return json_path, html_path
