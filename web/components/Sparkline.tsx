/**
 * A KPI tile's sparkline (`tm-kpi__spark`, design/tagg/charts.md § v1.1): the
 * chart rules in miniature — a 1.6 px line, a 12 % area under it, a dot on the
 * last point, nothing else. No axes, no hover: the full chart is the place for
 * those.
 *
 * Drawn in `currentColor`, so it takes the colour of the tile's number
 * (`tm-kpi--forest` and friends set it); a missing value breaks the line rather
 * than being read as zero.
 */

const WIDTH = 100;
const HEIGHT = 28;
// Room for the end dot, so it is not clipped at the edges.
const PAD = 2.5;

export function Sparkline({ values }: { values: (number | null | undefined)[] }) {
  const points = values
    .map((v, i) => (v == null || Number.isNaN(v) ? null : { i, v }))
    .filter((p): p is { i: number; v: number } => p !== null);
  if (points.length < 2) return null;

  const lo = Math.min(...points.map((p) => p.v));
  const hi = Math.max(...points.map((p) => p.v));
  const span = hi - lo || 1;
  const last = values.length - 1 || 1;
  const x = (i: number) => PAD + (i / last) * (WIDTH - 2 * PAD);
  const y = (v: number) => HEIGHT - PAD - ((v - lo) / span) * (HEIGHT - 2 * PAD);

  const line = points.map((p, k) => `${k ? "L" : "M"}${x(p.i)},${y(p.v)}`).join("");
  const first = points[0];
  const end = points[points.length - 1];
  const area = `${line}L${x(end.i)},${HEIGHT}L${x(first.i)},${HEIGHT}Z`;

  return (
    <span className="tm-kpi__spark" aria-hidden="true">
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} preserveAspectRatio="none">
        <path d={area} fill="currentColor" fillOpacity={0.12} stroke="none" />
        <path
          d={line}
          fill="none"
          stroke="currentColor"
          strokeWidth={1.6}
          strokeLinejoin="round"
          strokeLinecap="round"
          vectorEffect="non-scaling-stroke"
        />
        {/* A zero-length round-capped stroke, not a <circle>: the viewBox is
            stretched to the tile's width, and only a non-scaling stroke stays
            round under that. */}
        <path
          d={`M${x(end.i)},${y(end.v)}h0`}
          stroke="currentColor"
          strokeWidth={5}
          strokeLinecap="round"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
    </span>
  );
}
