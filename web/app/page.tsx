/**
 * Landing page for a visitor (design/tagg/visitor.md § Les trois surfaces), or
 * straight through to Home if already signed in.
 *
 * In this order: lockup and motto, one sentence, then the access grid — what is
 * open now, what opens with Strava, every entry a real link — and only then the
 * Strava button. The visitor sees what is theirs before being asked for anything.
 *
 * A server component so the session cookie decides before anything renders — no
 * flash of a sign-in screen for a signed-in user.
 */

import { redirect } from "next/navigation";

import { Icon, type IconName } from "@/components/Icon";
import { signInHref } from "@/lib/auth";
import { readSession } from "@/lib/session";
import { translator, type Translate } from "@/lib/strings";
import { loadStrings } from "@/lib/strings.server";

const OPEN: { href: string; label: string; desc: string; icon: IconName }[] = [
  { href: "/race-plan", label: "nav.race_plan", desc: "visitor.race_plan", icon: "flag" },
  { href: "/blog", label: "nav.blog", desc: "visitor.blog", icon: "newspaper" },
];

const WITH_STRAVA: typeof OPEN = [
  { href: "/home", label: "nav.home", desc: "visitor.home", icon: "home" },
  { href: "/pages", label: "nav.analysis", desc: "visitor.analysis", icon: "chart" },
  { href: "/training", label: "nav.training", desc: "visitor.training", icon: "calendar" },
];

export default async function Landing({
  searchParams,
}: {
  searchParams: Promise<{ error?: string }>;
}) {
  if (await readSession()) redirect("/home");
  const { error } = await searchParams;
  const t = translator(await loadStrings());

  return (
    <main className="container container--narrow landing">
      <div className="landing__brand">
        {/* eslint-disable-next-line @next/next/no-img-element -- a static SVG gains nothing from next/image */}
        <img src="/logo/tagg-lockup.svg" alt="TAGG" height={44} />
        <span className="kicker landing__motto">Train · Analyse · Guide · Grow</span>
      </div>

      <p className="lede">{t("visitor.lede")}</p>

      {error && <p className="note note--error">{error}</p>}

      <div className="tm-access">
        <AccessColumn
          title={t("visitor.open.title")}
          tier={t("visitor.open.tier")}
          tone="moss"
          items={OPEN}
          t={t}
        />
        <AccessColumn
          title={t("visitor.strava.title")}
          tier={t("visitor.strava.tier")}
          tone="forest"
          items={WITH_STRAVA}
          locked
          t={t}
        />
      </div>

      <div className="landing__connect">
        <a className="tm-btn tm-btn--strava" href={signInHref()}>
          {t("visitor.connect")}
        </a>
        <p className="body-sm landing__trust">{t("visitor.trust")}</p>
      </div>

      <p className="muted">
        <a href="/privacy">Privacy Policy</a> · <a href="/terms">Terms of Service</a>
      </p>
    </main>
  );
}

function AccessColumn({
  title,
  tier,
  tone,
  items,
  locked = false,
  t,
}: {
  title: string;
  tier: string;
  tone: "moss" | "forest";
  items: typeof OPEN;
  locked?: boolean;
  t: Translate;
}) {
  return (
    <div className="tm-access__col">
      <div className="tm-access__head">
        {title}
        <span className={`tm-chip tm-chip--${tone}`}>{tier}</span>
      </div>
      {items.map((item) => (
        <a
          key={item.href}
          className={`tm-access__item${locked ? " tm-access__item--locked" : ""}`}
          href={item.href}
        >
          <Icon name={item.icon} size={18} />
          <span>
            <span className="tm-access__title">{t(item.label)}</span>
            <br />
            <span className="tm-access__desc">{t(item.desc)}</span>
          </span>
        </a>
      ))}
    </div>
  );
}
