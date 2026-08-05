"use client";

/**
 * Ask for the athlete's email address.
 *
 * Strava's API returns no email under any scope, so the app has to ask — and the
 * only moment an athlete reliably passes through is straight after they connect,
 * which is why `/welcome` exists as its own step rather than as a banner someone can
 * scroll past.
 *
 * Validation is intentionally thin here: the shape check is the server's
 * (`api/routers/auth.py` owns the pattern), and duplicating the rule in the browser
 * would let the two drift. This only avoids a round-trip for an obviously empty box.
 */

import { useState } from "react";

import { updateProfile } from "@/lib/api";
import { translator, type Strings } from "@/lib/strings";

export function EmailForm({
  strings,
  initial = "",
  submitLabel,
  onSaved,
}: {
  strings: Strings;
  initial?: string;
  submitLabel?: string;
  /** Called with the saved address; the caller decides where to go next. */
  onSaved: (email: string) => void;
}) {
  const t = translator(strings);
  const [email, setEmail] = useState(initial);
  const [saving, setSaving] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    const value = email.trim();
    if (!value) {
      setFailure(t("email.invalid"));
      return;
    }
    setSaving(true);
    setFailure(null);
    try {
      await updateProfile({ email: value });
      onSaved(value);
    } catch {
      // Any rejection here is the pattern check: the endpoint is authenticated and
      // the only other input is a string.
      setFailure(t("email.invalid"));
    } finally {
      setSaving(false);
    }
  };

  return (
    <form className="email-form" onSubmit={submit}>
      <label className="email-form__label" htmlFor="athlete-email">
        {t("email.label")}
      </label>
      <div className="email-form__row">
        <input
          id="athlete-email"
          className="email-form__input"
          type="email"
          autoFocus
          autoComplete="email"
          required
          placeholder={t("email.placeholder")}
          value={email}
          onChange={(event) => setEmail(event.target.value)}
        />
        <button type="submit" className="button" disabled={saving}>
          {saving ? t("common.saving") : submitLabel ?? t("email.submit")}
        </button>
      </div>
      {failure && <p className="note note--error">{failure}</p>}
    </form>
  );
}
