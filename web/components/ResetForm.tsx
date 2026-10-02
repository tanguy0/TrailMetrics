"use client";

/**
 * "Forgot your password?" — both halves (design/specs/auth.md § Réinitialisation).
 *
 * Without a token: ask for the address and send a link, or, when no mail
 * provider is configured, say whom to write to instead (decided server-side and
 * passed in as `contact`). With a token: choose the new password, which signs
 * out every other device and signs this one in.
 */

import { useState, type FormEvent } from "react";

import { translator, type Strings } from "@/lib/strings";

export function ResetForm({
  token,
  contact,
  strings,
}: {
  token: string | null;
  contact: string | null;
  strings: Strings;
}) {
  const t = translator(strings);
  const [value, setValue] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState(false);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const response = await fetch(token ? "/api/auth/reset/confirm" : "/api/auth/reset", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(token ? { token, password: value } : { email: value }),
      });
      if (response.ok) {
        if (token) {
          window.location.assign("/home");
          return;
        }
        setSent(true);
      } else {
        const body = (await response.json().catch(() => ({}))) as { detail?: string | null };
        setError(body.detail || t("auth.error.generic"));
      }
    } catch {
      setError(t("auth.error.generic"));
    }
    setBusy(false);
  }

  const title = t(token ? "auth.reset.new_title" : "auth.reset.title");

  return (
    <form className="tm-auth" onSubmit={submit} noValidate>
      <h2 className="auth-page__title">{title}</h2>

      {!token && contact ? (
        <p className="body-sm">{t("auth.reset.write_to", { email: contact })}</p>
      ) : sent ? (
        <p className="body-sm">{t("auth.reset.sent")}</p>
      ) : (
        <>
          <p className="body-sm muted">{t(token ? "auth.reset.new_lede" : "auth.reset.lede")}</p>
          {error && <div className="tm-auth__error" role="alert">{error}</div>}
          <label className="tm-field">
            {t(token ? "auth.password" : "auth.email")}
            <input
              className="tm-input"
              type={token ? "password" : "email"}
              autoComplete={token ? "new-password" : "email"}
              required
              minLength={token ? 10 : undefined}
              maxLength={token ? 128 : 254}
              value={value}
              onChange={(event) => setValue(event.target.value)}
            />
          </label>
          {token && <div className="tm-auth__row"><span>{t("auth.password_hint")}</span></div>}
          <button className="tm-btn tm-btn--wide" type="submit" disabled={busy || !value}>
            {t(token ? "auth.reset.new_submit" : "auth.reset.submit")}
          </button>
        </>
      )}

      <p className="tm-auth__fine">
        <a href="/login">{t("auth.reset.back")}</a>
      </p>
    </form>
  );
}
