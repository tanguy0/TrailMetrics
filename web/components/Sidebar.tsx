"use client";

/**
 * The navigation rail — always on screen (design/tagg/components/NavRail.md).
 *
 * A client component only because the active item depends on the current path;
 * the labels arrive already translated from the server, and so does the viewer's
 * access tier, so nothing is fetched here.
 *
 * The rail follows the tier (design/tagg/access.md): a visitor sees what is open
 * now, then what opens with an account; an account without Strava sees its own
 * pages, then what opens with Strava. Items the tier does not open still link to
 * their page — which shows its teaser — and carry a lock rather than being greyed
 * out. On the auth pages the rail is reduced to the lockup.
 */

import { usePathname } from "next/navigation";

import { CoachSwitcher } from "@/components/CoachSwitcher";
import { Icon, type IconName } from "@/components/Icon";
import { loginHref } from "@/lib/auth";
import type { Viewer } from "@/lib/session";
import { translator, type Strings, type Translate } from "@/lib/strings";

type Tier = "visitor" | "account" | "strava";

interface Item {
  href: string;
  label: string;
  icon: IconName;
  needs: Tier;
}

const RANK: Record<Tier, number> = { visitor: 0, account: 1, strava: 2 };
const AUTH_PATHS = ["/login", "/register", "/reset", "/verify"];

async function signOut() {
  await fetch("/api/auth/logout", { method: "POST" }).catch(() => undefined);
  window.location.assign("/");
}

export function Sidebar({ strings, viewer }: { strings: Strings; viewer: Viewer | null }) {
  const t = translator(strings);
  const pathname = usePathname() ?? "";
  const tier: Tier = viewer?.tier ?? "visitor";

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
    { href: "/pages", label: t("nav.analysis"), icon: "chart", needs: "strava" },
    { href: "/training", label: t("nav.training"), icon: "calendar", needs: "strava" },
    { href: "/race-plan", label: t("nav.race_plan"), icon: "flag", needs: "visitor" },
    { href: "/blog", label: t("nav.blog"), icon: "newspaper", needs: "visitor" },
  ];
  const open = items.filter((item) => RANK[item.needs] <= RANK[tier]);
  const locked = items.filter((item) => RANK[item.needs] > RANK[tier]);

  return (
    <nav className="tm-rail shell__rail" aria-label={t("nav.analysis")}>
      {brand}

      {viewer?.isCoach && <CoachSwitcher />}

      {locked.length === 0 ? (
        <RailList items={open} pathname={pathname} t={t} />
      ) : (
        <div className="shell__groups">
          {tier === "visitor" && <div className="tm-rail__group">{t("nav.group_open")}</div>}
          <RailList items={open} pathname={pathname} t={t} />
          <div className="tm-rail__group">
            {t(tier === "visitor" ? "nav.group_account" : "nav.group_strava")}
          </div>
          <RailList items={locked} pathname={pathname} locked t={t} />
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
