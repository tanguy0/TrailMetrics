"use client";

/**
 * A coach account's switch into another athlete's account.
 *
 * Renders nothing for anyone whose signed-in identity is not in
 * `COACH_ATHLETE_IDS` — `athlete.is_coach` reports on the real, signed-in
 * account, not whichever one is currently being viewed. The switch itself is a
 * cookie (see `/api/view-as`); the actual access control happens server-side on
 * every request (`api/deps.py`'s `current_athlete_id`), so this component is
 * just a convenience, not something that has to be trusted.
 */

import { useEffect, useState } from "react";

import { getAthlete, listCoachAthletes } from "@/lib/api";
import type { Athlete, CoachAthlete } from "@/lib/types";

async function switchTo(athleteId: number | null) {
  await fetch("/api/view-as", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ athleteId }),
  });
  window.location.href = "/home";
}

export function CoachSwitcher() {
  const [athlete, setAthlete] = useState<Athlete | null>(null);
  const [roster, setRoster] = useState<CoachAthlete[] | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    getAthlete().then(setAthlete).catch(() => {});
  }, []);

  if (!athlete?.is_coach) return null;

  const toggle = async () => {
    if (!open && roster === null) {
      try {
        setRoster((await listCoachAthletes()).athletes);
      } catch {
        setRoster([]);
      }
    }
    setOpen((value) => !value);
  };

  return (
    <div className="coach-switcher">
      <button
        type="button"
        className={
          "coach-switcher__toggle" +
          (athlete.viewing_as ? " coach-switcher__toggle--active" : "")
        }
        onClick={toggle}
      >
        <span aria-hidden="true">{athlete.viewing_as ? "👁️" : "🧑‍🤝‍🧑"}</span>
        <span>{athlete.viewing_as ? `Viewing: ${athlete.display_name}` : "Switch athlete"}</span>
      </button>

      {open && (
        <ul className="coach-switcher__list">
          {athlete.viewing_as && (
            <li>
              <button type="button" onClick={() => switchTo(null)}>
                ← Back to my account
              </button>
            </li>
          )}
          {roster === null ? (
            <li className="coach-switcher__hint">Loading…</li>
          ) : roster.length === 0 ? (
            <li className="coach-switcher__hint">No other athletes yet.</li>
          ) : (
            roster.map((entry) => (
              <li key={entry.id}>
                <button type="button" onClick={() => switchTo(entry.id)}>
                  {entry.display_name}
                </button>
              </li>
            ))
          )}
        </ul>
      )}
    </div>
  );
}
