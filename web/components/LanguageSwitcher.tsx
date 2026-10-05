"use client";

/**
 * The athlete's UI-language choice — English or French — top right of every
 * signed-in page.
 *
 * Two writes on click, mirroring `CoachSwitcher`'s split between durable state
 * and this-browser state: `updateProfile({ lang })` persists it to the athlete's
 * row so it follows them to another device, and `/api/lang` sets the cookie
 * `lib/session.ts`'s `lang()` reads synchronously for the server-rendered shell.
 * A full reload (not `router.refresh()`) afterwards, because the registry and
 * UI-strings caches in `lib/api.ts` are cached in module state for the tab's
 * lifetime and a soft refresh would leave them stale.
 *
 * The highlighted option is `current` — the language the page was actually
 * rendered in (the cookie, read by the layout) — not the stored preference,
 * which can differ (an account's language vs its Strava athlete's row) and then
 * lit EN on a French page.
 */

import { useEffect, useState } from "react";

import { getAthlete, updateProfile } from "@/lib/api";
import type { Athlete } from "@/lib/types";

const LANGUAGES: { code: "en" | "fr"; label: string }[] = [
  { code: "en", label: "EN" },
  { code: "fr", label: "FR" },
];

export function LanguageSwitcher({ current }: { current: string }) {
  const [athlete, setAthlete] = useState<Athlete | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getAthlete().then(setAthlete).catch(() => {});
  }, []);

  if (!athlete) return null;

  const choose = async (code: "en" | "fr") => {
    if (code === current || busy) return;
    setBusy(true);
    try {
      await Promise.all([
        updateProfile({ lang: code }),
        fetch("/api/lang", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ lang: code }),
        }),
      ]);
      window.location.reload();
    } catch {
      // A failed switch leaves the toggle clickable again rather than stuck busy.
      setBusy(false);
    }
  };

  return (
    <div className="lang-switch" role="group" aria-label="Language">
      {LANGUAGES.map(({ code, label }) => (
        <button
          key={code}
          type="button"
          className={
            "lang-switch__option" +
            (current === code ? " lang-switch__option--active" : "")
          }
          aria-pressed={current === code}
          disabled={busy}
          onClick={() => choose(code)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
