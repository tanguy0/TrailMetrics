"use client";

/**
 * The app's progress bar.
 *
 * A `<progress>` element renders as whatever the OS feels like — a grey pill on
 * macOS, something else on Windows — and is close to unstyleable. This is two divs
 * instead, so a long import looks like it belongs to the same product as the charts
 * beside it, in the same palette.
 *
 * It handles the case the native element handles badly: **unknown total**. A sync
 * that is still listing activities knows only that it is working, so the bar sweeps
 * rather than claiming 0%. That distinction is the whole reason this exists — a bar
 * frozen at zero reads as broken.
 */

interface Props {
  /** Units finished. Ignored when `total` is 0. */
  value: number;
  /** Units expected; 0 or less means "not known yet" → indeterminate. */
  total: number;
  /** Short line above the bar: what is happening. */
  label?: string;
  /** Short line beside the count: extra detail from the server. */
  detail?: string;
  tone?: "forest" | "sunrise";
}

export function ProgressBar({ value, total, label, detail, tone = "forest" }: Props) {
  const determinate = total > 0;
  const done = determinate ? Math.min(Math.max(value, 0), total) : 0;
  const percent = determinate ? Math.round((done / total) * 100) : null;

  return (
    <div className={`progress-bar progress-bar--${tone}`}>
      {(label || percent != null) && (
        <div className="progress-bar__head">
          {label && <span className="progress-bar__label">{label}</span>}
          {percent != null && (
            <span className="progress-bar__count">
              {done} / {total}
              {/* An explicit separator: "0 / 7" followed by "0%" reads as "0 / 70%"
                  the moment the two are adjacent, and spacing alone does not survive
                  being copied out of the page. */}
              <span aria-hidden="true"> · </span>
              <span className="progress-bar__percent">{percent}%</span>
            </span>
          )}
        </div>
      )}

      <div
        className={`progress-bar__track${
          determinate ? "" : " progress-bar__track--indeterminate"
        }`}
        // Announced as a progress bar either way; an indeterminate one deliberately
        // reports no value, which is what tells a screen reader "working, unknown".
        role="progressbar"
        aria-valuemin={determinate ? 0 : undefined}
        aria-valuemax={determinate ? total : undefined}
        aria-valuenow={determinate ? done : undefined}
        aria-label={label}
      >
        <div
          className="progress-bar__fill"
          style={determinate ? { width: `${percent}%` } : undefined}
        />
      </div>

      {detail && <span className="progress-bar__detail">{detail}</span>}
    </div>
  );
}
