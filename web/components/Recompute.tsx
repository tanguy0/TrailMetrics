"use client";

/**
 * When a tool's result was computed, and the button that computes it again.
 *
 * Profiles and saved race plans are kept as last computed rather than refitted on
 * every visit: this is the one place an athlete sees how old they are (and how
 * many runs came in since) and asks for a fresh fit.
 */

import { formatDate } from "@/lib/format";
import { plural, type Translate } from "@/lib/strings";

export function Recompute({
  computedAt,
  newRuns = 0,
  busy,
  onRecompute,
  t,
}: {
  computedAt: string | null | undefined;
  /** Runs newer than the fit; unknown (a race plan) counts as none. */
  newRuns?: number;
  busy: boolean;
  onRecompute: () => void;
  t: Translate;
}) {
  return (
    <div className="recompute">
      {busy ? (
        <span className="pending">
          <span className="spinner" aria-hidden="true" />
          <span className="muted">{t("recompute.busy")}</span>
        </span>
      ) : (
        computedAt && (
          <span className="muted recompute__when">
            {t("recompute.computed", { when: formatDate(computedAt, "relative", t("locale")) })}
            {newRuns > 0 && ` · ${plural(t, "recompute.new_runs", newRuns)}`}
          </span>
        )
      )}
      <button
        type="button"
        className="tm-btn tm-btn--secondary tm-btn--sm"
        onClick={onRecompute}
        disabled={busy}
      >
        {t("recompute.button")}
      </button>
    </div>
  );
}
