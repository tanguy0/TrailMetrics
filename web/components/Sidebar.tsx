"use client";

/**
 * The navigation rail — always on screen, signed in or not
 * (design/tagg/components/NavRail.md).
 *
 * A client component only because the active item depends on the current path;
 * the labels arrive already translated from the server, so nothing is fetched here.
 * Every item but Race plan and Blog needs an athlete's own data, so `authenticated` renders
 * those as inert (no href, no click) rather than hiding the rail itself — a
 * visitor should see what TAGG offers before signing in, not guess.
 */

import { usePathname } from "next/navigation";

import { CoachSwitcher } from "@/components/CoachSwitcher";
import { Icon, type IconName } from "@/components/Icon";
import { translator, type Strings } from "@/lib/strings";

export function Sidebar({
  strings,
  authenticated,
}: {
  strings: Strings;
  authenticated: boolean;
}) {
  const t = translator(strings);
  const pathname = usePathname() ?? "";

  const items: { href: string; label: string; icon: IconName; public: boolean }[] = [
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

      <ul className="tm-rail__nav">
        {items.map((item) => {
          // `startsWith` so a page being edited (/pages/abc) keeps its tab lit.
          const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
          const enabled = authenticated || item.public;

          if (!enabled) {
            return (
              <li key={item.href}>
                <span
                  className="tm-rail__link tm-rail__link--disabled"
                  aria-disabled="true"
                  title={t("nav.sign_in_required")}
                >
                  <Icon name={item.icon} size={17} />
                  <span>{item.label}</span>
                </span>
              </li>
            );
          }

          return (
            <li key={item.href}>
              <a
                className={`tm-rail__link${active ? " tm-rail__link--active" : ""}`}
                href={item.href}
                aria-current={active ? "page" : undefined}
              >
                <Icon name={item.icon} size={17} />
                <span>{item.label}</span>
              </a>
            </li>
          );
        })}
      </ul>

      {authenticated && (
        <a className="tm-rail__link shell__signout" href="/api/auth/logout">
          <Icon name="logout" size={17} />
          <span>{t("nav.sign_out")}</span>
        </a>
      )}
    </nav>
  );
}
