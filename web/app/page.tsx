/**
 * Landing page for a visitor (design/tagg/access.md § Arrivée sur le site), or
 * straight through to Home if already signed in.
 *
 * In this order: lockup and motto, one sentence, then the access grid — what is
 * open now, what opens with an account, every entry a real link — and only then
 * the two buttons, Create an account first. The visitor sees what is theirs
 * before being asked for anything. Strava is not here any more: it attaches to
 * an account, from Home.
 *
 * A server component so the session cookie decides before anything renders — no
 * flash of a sign-in screen for a signed-in user.
 */

import { redirect } from "next/navigation";

import { Icon, type IconName } from "@/components/Icon";
import { Callout } from "@/components/Callout";
import { loginHref, registerHref } from "@/lib/auth";
import { getViewer } from "@/lib/session";
import { translator, type Translate } from "@/lib/strings";
import { loadStrings } from "@/lib/strings.server";

const OPEN: { href: string; label: string; desc: string; icon: IconName }[] = [
  { href: "/tools/race-planning", label: "tools.race_planning", desc: "visitor.race_plan", icon: "flag" },
  { href: "/tools/level", label: "tools.level", desc: "visitor.level", icon: "target" },
  { href: "/blog", label: "nav.blog", desc: "visitor.blog", icon: "newspaper" },
];

const WITH_ACCOUNT: typeof OPEN = [
  { href: "/home", label: "nav.home", desc: "visitor.home", icon: "home" },
  { href: "/tools/gap", label: "nav.tools", desc: "visitor.tools", icon: "ruler" },
  { href: "/coaching", label: "nav.coaching", desc: "visitor.coaching", icon: "calendar" },
];

export default async function Landing({
  searchParams,
}: {
  searchParams: Promise<{ error?: string }>;
}) {
  if (await getViewer()) redirect("/home");
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

      {error && <Callout tone="terra">{error}</Callout>}

      <div className="tm-access">
        <AccessColumn
          title={t("visitor.open.title")}
          tier={t("visitor.open.tier")}
          tone="moss"
          items={OPEN}
          t={t}
        />
        <AccessColumn
          title={t("visitor.account.title")}
          tier={t("visitor.account.tier")}
          tone="forest"
          items={WITH_ACCOUNT}
          locked
          t={t}
        />
      </div>

      <div className="landing__connect">
        <div className="landing__actions">
          <a className="tm-btn" href={registerHref()}>
            {t("visitor.register")}
          </a>
          <a className="tm-btn tm-btn--secondary" href={loginHref()}>
            {t("visitor.login")}
          </a>
        </div>
        <p className="body-sm landing__trust">{t("visitor.account_trust")}</p>
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
    // "With an account" is the landing's one tinted block (contrast.md § 1): the
    // column the visitor is being invited into.
    <div className={`tm-access__col landing__col${locked ? " tm-panel--tint" : ""}`}>
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
