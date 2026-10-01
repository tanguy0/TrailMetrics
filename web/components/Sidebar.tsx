"use client";

/**
 * The navigation rail — always on screen, signed in or not
 * (design/tagg/components/NavRail.md).
 *
 * A client component only because the active item depends on the current path;
 * the labels arrive already translated from the server, so nothing is fetched here.
 *
 * Signed in, it is one list. For a visitor it is two groups (visitor.md § Rail):
 * what is open now (Race plan, Blog), then what opens with Strava — those still
 * link to their page, which shows its teaser, and carry a lock rather than being
 * greyed out — with the Strava button at the foot of the rail.
 */

import { usePathname } from "next/navigation";

import { CoachSwitcher } from "@/components/CoachSwitcher";
import { Icon, type IconName } from "@/components/Icon";
import { signInHref } from "@/lib/auth";
import { translator, type Strings, type Translate } from "@/lib/strings";

interface Item {
  href: string;
  label: string;
  icon: IconName;
  public: boolean;
}

export function Sidebar({
  strings,
  authenticated,
}: {
  strings: Strings;
  authenticated: boolean;
}) {
  const t = translator(strings);
  const pathname = usePathname() ?? "";

  const items: Item[] = [
    { href: "/home", label: t("nav.home"), icon: "home", public: false },
    { href: "/pages", label: t("nav.analysis"), icon: "chart", public: false },
    { href: "/training", label: t("nav.training"), icon: "calendar", public: false },
    { href: "/race-plan", label: t("nav.race_plan"), icon: "flag", public: true },
    { href: "/blog", label: t("nav.blog"), icon: "newspaper", public: true },
  ];

  return (
    <nav className="tm-rail shell__rail" aria-label={t("nav.analysis")}>
      <a className="tm-rail__brand" href={authenticated ? "/home" : "/"}>
        {/* eslint-disable-next-line @next/next/no-img-element -- a static SVG gains nothing from next/image */}
        <img src="/logo/tagg-lockup-on-rail.svg" alt="TAGG" height={28} />
      </a>

      {authenticated && <CoachSwitcher />}

      {authenticated ? (
        <RailList items={items} pathname={pathname} t={t} />
      ) : (
        <div className="shell__groups">
          <div className="tm-rail__group">{t("nav.group_open")}</div>
          <RailList items={items.filter((item) => item.public)} pathname={pathname} t={t} />
          <div className="tm-rail__group">{t("nav.group_strava")}</div>
          <RailList items={items.filter((item) => !item.public)} pathname={pathname} locked t={t} />
        </div>
      )}

      {authenticated ? (
        <a className="tm-rail__link shell__signout" href="/api/auth/logout">
          <Icon name="logout" size={17} />
          <span>{t("nav.sign_out")}</span>
        </a>
      ) : (
        <a
          className="tm-btn tm-btn--strava tm-btn--sm tm-btn--wide shell__signout"
          href={signInHref(pathname || undefined)}
        >
          {t("nav.connect")}
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
