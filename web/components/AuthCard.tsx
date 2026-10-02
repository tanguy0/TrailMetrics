"use client";

/**
 * Sign in / create an account (`tm-auth`, design/specs/auth.md § Pages web).
 *
 * One card, two modes behind a `tm-segment`. The segment items are links, not
 * tabs that swap state: each mode has its own URL (/login, /register), so the
 * browser's back button and a shared link both do the obvious thing.
 *
 * The form posts with `fetch` to this app's own `/api/auth/*` routes, which set
 * the session cookie; the token never reaches this component. On success the
 * page is left with a full navigation, so the server-rendered shell (rail, tier)
 * is rebuilt for the signed-in account.
 */

import Link from "next/link";
import { useState, type FormEvent } from "react";

import { loginHref, registerHref } from "@/lib/auth";
import { translator, type Strings } from "@/lib/strings";

export function AuthCard({
  mode,
  next,
  strings,
}: {
  mode: "login" | "register";
  next: string;
  strings: Strings;
}) {
  const t = translator(strings);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const response = await fetch(`/api/auth/${mode}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      if (response.ok) {
        window.location.assign(next);
        return;
      }
      const body = (await response.json().catch(() => ({}))) as { detail?: string | null };
      setError(body.detail || t("auth.error.generic"));
    } catch {
      setError(t("auth.error.generic"));
    }
    setBusy(false);
  }

  return (
    <form className="tm-auth" onSubmit={submit} noValidate>
      <div className="tm-segment" role="tablist">
        <a
          className="tm-segment__item"
          role="tab"
          aria-selected={mode === "login"}
          href={loginHref(next === "/home" ? undefined : next)}
        >
          {t("auth.login.title")}
        </a>
        <a
          className="tm-segment__item"
          role="tab"
          aria-selected={mode === "register"}
          href={registerHref(next === "/home" ? undefined : next)}
        >
          {t("auth.register.title")}
        </a>
      </div>

      {error && <div className="tm-auth__error" role="alert">{error}</div>}

      <label className="tm-field">
        {t("auth.email")}
        <input
          className="tm-input"
          type="email"
          name="email"
          autoComplete="email"
          required
          value={email}
          onChange={(event) => setEmail(event.target.value)}
        />
      </label>

      <label className="tm-field">
        {t("auth.password")}
        <input
          className="tm-input"
          type="password"
          name="password"
          autoComplete={mode === "login" ? "current-password" : "new-password"}
          required
          minLength={mode === "register" ? 10 : undefined}
          maxLength={128}
          value={password}
          onChange={(event) => setPassword(event.target.value)}
        />
      </label>

      <div className="tm-auth__row">
        {mode === "register" ? <span>{t("auth.password_hint")}</span> : <span />}
        {mode === "login" && <Link href="/reset">{t("auth.forgot")}</Link>}
      </div>

      <button className="tm-btn tm-btn--wide" type="submit" disabled={busy || !email || !password}>
        {t(mode === "login" ? "auth.submit.login" : "auth.submit.register")}
      </button>

      {mode === "register" && (
        <p className="tm-auth__fine">
          {t("auth.fine")} <a href="/terms">Terms</a> · <a href="/privacy">Privacy</a>
        </p>
      )}
    </form>
  );
}
