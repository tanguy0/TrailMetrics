/**
 * Dormant while mail is off (api/mail.py): no verification link is sent, so
 * nothing links here until MAIL_FROM is configured.
 *
 * The page a verification link opens. Confirms on render, server-side: the token
 * is the proof, so no session is needed — the link may be opened on another
 * device than the one signed in. A link scanner opening it first confirms the
 * address just the same, which is what the link is for.
 */

import type { Metadata } from "next";

import { apiBaseUrl, lang } from "@/lib/session";
import { translator } from "@/lib/strings";
import { loadStrings } from "@/lib/strings.server";

export const metadata: Metadata = {
  title: "TAGG",
  // The token is in the URL: keep it out of any Referer this page could send.
  referrer: "no-referrer",
};

async function confirm(token: string): Promise<{ email: string } | { detail: string | null }> {
  try {
    const response = await fetch(
      `${apiBaseUrl()}/auth/verify/confirm?lang=${encodeURIComponent(await lang())}`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ token }),
        cache: "no-store",
      },
    );
    const body = (await response.json().catch(() => ({}))) as { email?: string; detail?: unknown };
    if (response.ok && body.email) return { email: body.email };
    return { detail: typeof body.detail === "string" ? body.detail : null };
  } catch {
    return { detail: null };
  }
}

export default async function VerifyPage({ params }: { params: Promise<{ token: string }> }) {
  const t = translator(await loadStrings());
  const result = await confirm((await params).token);
  const ok = "email" in result;

  return (
    <main className="container auth-page">
      <div className="tm-auth">
        <h2 className="auth-page__title">
          {t(ok ? "auth.verify.done.title" : "auth.verify.failed.title")}
        </h2>
        {ok ? (
          <p className="body-sm">{t("auth.verify.done.body", { email: result.email })}</p>
        ) : (
          <div className="tm-auth__error" role="alert">
            {result.detail || t("auth.error.generic")}
          </div>
        )}
        <a className="tm-btn tm-btn--wide" href="/home">
          {t("auth.verify.continue")}
        </a>
      </div>
    </main>
  );
}
