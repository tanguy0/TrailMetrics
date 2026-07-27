"use client";

/**
 * TableData → an HTML table, with per-column formatting, best-value highlighting
 * and a CSV export of the *unformatted* values.
 *
 * The server sends numbers and a format descriptor rather than strings, so the same
 * payload can be displayed, sorted and exported without re-parsing anything.
 */

import { useMemo } from "react";

import { downloadCsv, formatCell, toCsv } from "@/lib/format";
import type { Column, TableData } from "@/lib/types";

/** Row indices holding the best value in each highlighted column. */
function bestRows(table: TableData): Map<string, Set<number>> {
  const best = new Map<string, Set<number>>();
  for (const column of table.columns) {
    if (!column.highlight || table.rows.length < 2) continue;
    let target: number | null = null;
    for (const row of table.rows) {
      const value = Number(row[column.key]);
      if (!Number.isFinite(value)) continue;
      if (target == null) target = value;
      else target = column.highlight === "max" ? Math.max(target, value) : Math.min(target, value);
    }
    if (target == null) continue;
    const winners = new Set<number>();
    table.rows.forEach((row, index) => {
      if (Number(row[column.key]) === target) winners.add(index);
    });
    best.set(column.key, winners);
  }
  return best;
}

export function TableView({ table }: { table: TableData }) {
  const best = useMemo(() => bestRows(table), [table]);

  if (!table.rows.length) return null;

  const exportCsv = () => {
    const headers = table.columns.map((c: Column) => c.label);
    const rows = table.rows.map((row) => table.columns.map((c) => row[c.key] ?? ""));
    downloadCsv(table.download_name, toCsv(headers, rows));
  };

  return (
    <div className="table-block">
      {table.title && <h4 className="table-block__title">{table.title}</h4>}
      <div className="table-scroll">
        <table className="table">
          <thead>
            <tr>
              {table.columns.map((column) => (
                <th key={column.key}>{column.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {table.columns.map((column) => (
                  <td
                    key={column.key}
                    className={best.get(column.key)?.has(rowIndex) ? "cell--best" : undefined}
                  >
                    {formatCell(row[column.key], column.format)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {table.caption && <p className="muted">{table.caption}</p>}
      <button type="button" className="button button--ghost button--small" onClick={exportCsv}>
        Download table (CSV)
      </button>
    </div>
  );
}
