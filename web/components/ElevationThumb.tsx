/**
 * A course's elevation profile at thumbnail size, for the saved race plans list.
 *
 * The race plan's profile in miniature (design/tagg/charts.md § Plan de course):
 * a flat `line-strong` area at 35 %, no axes, no hover — the plan itself is the
 * place for those. The lowest and highest points are printed in mono beside it so
 * the shape still says how much it climbs.
 */

import { formatNumber } from "@/lib/format";

const WIDTH = 100;
const HEIGHT = 40;

export function ElevationThumb({ profile }: { profile: [number, number][] }) {
  if (profile.length < 2) return null;

  const elevations = profile.map(([, m]) => m);
  const lo = Math.min(...elevations);
  const hi = Math.max(...elevations);
  // A flat course still reads as a course, not as a line glued to the bottom.
  const span = Math.max(hi - lo, 50);
  const total = profile[profile.length - 1][0] || 1;
  const x = (km: number) => (km / total) * WIDTH;
  const y = (m: number) => HEIGHT - ((m - lo) / span) * (HEIGHT - 2);

  const top = profile.map(([km, m], k) => `${k ? "L" : "M"}${x(km)},${y(m)}`).join("");
  const area = `${top}L${WIDTH},${HEIGHT}L0,${HEIGHT}Z`;

  return (
    <div className="elevation-thumb" aria-hidden="true">
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} preserveAspectRatio="none">
        <path d={area} fill="var(--line-strong)" fillOpacity={0.35} stroke="none" />
      </svg>
      <span className="elevation-thumb__range">
        {formatNumber(lo, 0)}–{formatNumber(hi, 0)}
        <span className="elevation-thumb__unit"> m</span>
      </span>
    </div>
  );
}
