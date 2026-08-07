"use client";

/**
 * The first-run step: collect the email address, then get out of the way.
 *
 * A client component only so it can navigate after the save. Where it goes next is
 * carried in from the OAuth `state`, so connecting from a shared page link still ends
 * up on that page.
 */

import { useRouter } from "next/navigation";

import { EmailForm } from "@/components/EmailForm";
import { translator, type Strings } from "@/lib/strings";

export function WelcomeScreen({
  strings,
  next,
}: {
  strings: Strings;
  next: string;
}) {
  const t = translator(strings);
  const router = useRouter();

  return (
    <main className="container container--narrow">
      <section className="card-block card-block--welcome">
        <h1 className="welcome__title">{t("email.title")}</h1>
        <p className="lede">{t("email.body")}</p>
        <EmailForm strings={strings} onSaved={() => router.replace(next)} />
        <p className="muted">
          If you're being coached through this app, your coach's account can see your
          training data and training diary to help plan your training — see the{" "}
          <a href="/privacy">Privacy Policy</a> for what that does and doesn't include.
        </p>
      </section>
    </main>
  );
}
