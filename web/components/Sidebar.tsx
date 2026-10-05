"use client";

/**
 * The navigation rail — always on screen (design/tagg/components/NavRail.md).
 *
 * A client component only because the active item depends on the current path;
 * the labels arrive already translated from the server, and so does the viewer's
 * access tier, so nothing is fetched here.
 *
 * The rail follows the tier (design/tagg/access.md § Navigation): Home · Tools ·
 * Analysis · Coaching · Blog. What the viewer's tier opens comes first; what it
 * does not is grouped under the tier that opens it ("With an account", "With
 * Strava", "Coached by TAGG"). Those items still link to their page — which shows
 * its teaser — and carry a lock rather than being greyed out. On the auth pages
 * the rail is reduced to the lockup.
 */

import { usePathname } from "next/navigation";

import { CoachRequests } from "@/components/CoachRequests";
import { CoachSwitcher } from "@/components/CoachSwitcher";
import { Icon, type IconName } from "@/components/Icon";
import { loginHref } from "@/lib/auth";
import type { Viewer } from "@/lib/session";
import { translator, type Strings, type Translate } from "@/lib/strings";

type Tier = "visitor" | "account" | "strava" | "coached";

interface Item {
  href: string;
  label: string;
  icon: IconName;
  needs: Tier;
}

const GROUP: Record<Exclude<Tier, "visitor">, string> = {
  account: "nav.group_account",
  strava: "nav.group_strava",
  coached: "nav.group_coached",
};
const AUTH_PATHS = ["/login", "/register", "/reset", "/verify"];

async function signOut() {
  await fetch("/api/auth/logout", { method: "POST" }).catch(() => undefined);
  window.location.assign("/");
}

export function Sidebar({ strings, viewer }: { strings: Strings; viewer: Viewer | null }) {
  const t = translator(strings);
  const pathname = usePathname() ?? "";
  const tier = viewer?.tier ?? "visitor";
  // Not a ladder: coaching is a service, so a coached account without Strava
  // still opens Coaching, and a coach opens it like a coached athlete (their own
  // diary; requests arrive on the rail, under the switcher).
  const opens = (needs: Tier) =>
    needs === "visitor" ||
    (needs === "account" && viewer != null) ||
    (needs === "strava" && tier === "strava") ||
    (needs === "coached" && Boolean(viewer?.isCoached || viewer?.isCoach));

  const brand = (
    <a className="tm-rail__brand" href={viewer ? "/home" : "/"}>
      {/* eslint-disable-next-line @next/next/no-img-element -- a static SVG gains nothing from next/image */}
      <img src="/logo/tagg-lockup-on-rail.svg" alt="TAGG" height={28} />
    </a>
  );

  if (AUTH_PATHS.some((path) => pathname === path || pathname.startsWith(`${path}/`))) {
    return <nav className="tm-rail shell__rail shell__rail--bare" aria-label="TAGG">{brand}</nav>;
  }

  const items: Item[] = [
    { href: "/home", label: t("nav.home"), icon: "home", needs: "account" },
    { href: "/tools", label: t("nav.tools"), icon: "ruler", needs: "visitor" },
    { href: "/pages", label: t("nav.analysis"), icon: "chart", needs: "strava" },
    { href: "/coaching", label: t("nav.coaching"), icon: "calendar", needs: "coached" },
    { href: "/blog", label: t("nav.blog"), icon: "newspaper", needs: "visitor" },
  ];
  const open = items.filter((item) => opens(item.needs));
  const lockedTiers = (["account", "strava", "coached"] as const).filter(
    (needs) => !opens(needs) && items.some((item) => item.needs === needs),
  );

  return (
    <nav className="tm-rail shell__rail" aria-label={t("nav.analysis")}>
      {brand}

      {viewer?.isCoach && (
        <div className="shell__coach">
          <CoachSwitcher t={t} />
          <CoachRequests t={t} />
        </div>
      )}

      {lockedTiers.length === 0 ? (
        <RailList items={open} pathname={pathname} t={t} />
      ) : (
        <div className="shell__groups">
          {tier === "visitor" && <div className="tm-rail__group">{t("nav.group_open")}</div>}
          <RailList items={open} pathname={pathname} t={t} />
          {lockedTiers.map((needs) => (
            <div key={needs}>
              <div className="tm-rail__group">{t(GROUP[needs])}</div>
              <RailList
                items={items.filter((item) => item.needs === needs)}
                pathname={pathname}
                locked
                t={t}
              />
            </div>
          ))}
        </div>
      )}

      {viewer ? (
        <button type="button" className="tm-rail__link shell__signout" onClick={signOut}>
          <Icon name="logout" size={17} />
          <span>{t("nav.sign_out")}</span>
        </button>
      ) : (
        <a
          className="tm-btn tm-btn--secondary tm-btn--sm tm-btn--wide shell__signout"
          href={loginHref(pathname && pathname !== "/" ? pathname : undefined)}
        >
          {t("visitor.login")}
        </a>
      )}
    </nav>
  );
}

function RailList({
  items,
  pathname,
  locked = false,
  t,
}: {
  items: Item[];
  pathname: string;
  locked?: boolean;
  t: Translate;
}) {
  return (
    <ul className="tm-rail__nav">
      {items.map((item) => {
        // `startsWith` so a page being edited (/pages/abc) keeps its tab lit.
        const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
        return (
          <li key={item.href}>
            <a
              className={
                "tm-rail__link" +
                (active ? " tm-rail__link--active" : "") +
                (locked ? " tm-rail__link--locked" : "")
              }
              href={item.href}
              aria-current={active ? "page" : undefined}
              title={locked ? t("nav.sign_in_required") : undefined}
            >
              <Icon name={item.icon} size={17} />
              <span>{item.label}</span>
              {locked && <Icon name="lock" size={14} className="tm-lock" />}
            </a>
          </li>
        );
      })}
    </ul>
  );
}
