/**
 * A KPI tile (`tm-kpi`, design/tagg/components/KpiTile.md) for the Tools pages.
 * The number in mono, its unit muted and smaller; `tone` colours the number in
 * the card's role — one headline tile per card.
 */

import { kpiNumClass } from "@/lib/format";

export function Kpi({
  label,
  value,
  unit,
  note,
  tone,
  large = false,
}: {
  label: string;
  value: string;
  unit?: string;
  note?: string | null;
  tone?: "forest" | "terra" | "sun" | "moss";
  large?: boolean;
}) {
  const num = large && value.length <= 5 ? "tm-kpi__num tm-kpi__num--lg" : kpiNumClass(value);
  return (
    <div className={`tm-kpi tm-kpi--flat${tone ? ` tm-kpi--${tone}` : ""}`}>
      <span className="tm-kpi__label">{label}</span>
      <span className="tm-kpi__value">
        <span className={num}>{value}</span>
        {unit && value !== "—" && <span className="tm-kpi__unit">{unit}</span>}
      </span>
      {note && <span className="kpi__note">{note}</span>}
    </div>
  );
}
