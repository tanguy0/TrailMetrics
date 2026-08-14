"use client";

/**
 * The navigation rail — always on screen, signed in or not.
 *
 * A client component only because the active item depends on the current path;
 * the labels arrive already translated from the server, so nothing is fetched here.
 * Every item but Blog needs an athlete's own data, so `authenticated` renders
 * those as inert (no href, no click) rather than hiding the rail itself — a
 * visitor should see what TrailMetrics offers before signing in, not guess.
 */

import { usePathname } from "next/navigation";

import { CoachSwitcher } from "@/components/CoachSwitcher";
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

  const items = [
    { href: "/home", label: t("nav.home"), icon: "🏠", public: false },
    { href: "/pages", label: t("nav.analysis"), icon: "📊", public: false },
    { href: "/training", label: t("nav.training"), icon: "📅", public: false },
    { href: "/blog", label: t("nav.blog"), icon: "📰", public: true },
  ];

  return (
    <nav className="sidebar" aria-label={t("nav.analysis")}>
      <a className="sidebar__brand" href={authenticated ? "/home" : "/"}>
        <img className="sidebar__brand-logo" src="/logo.webp" alt="TrailMetrics" />
      </a>

      {authenticated && <CoachSwitcher />}

      <ul className="sidebar__nav">
        {items.map((item) => {
          // `startsWith` so a page being edited (/pages/abc) keeps its tab lit.
          const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
          const enabled = authenticated || item.public;

          if (!enabled) {
            return (
              <li key={item.href}>
                <span
                  className="sidebar__link sidebar__link--disabled"
                  aria-disabled="true"
                  title={t("nav.sign_in_required")}
                >
                  <span className="sidebar__icon" aria-hidden="true">{item.icon}</span>
                  <span>{item.label}</span>
                </span>
              </li>
            );
          }

          return (
            <li key={item.href}>
              <a
                className={`sidebar__link${active ? " sidebar__link--active" : ""}`}
                href={item.href}
                aria-current={active ? "page" : undefined}
              >
                <span className="sidebar__icon" aria-hidden="true">{item.icon}</span>
                <span>{item.label}</span>
              </a>
            </li>
          );
        })}
      </ul>

      {authenticated && (
        <a className="sidebar__signout" href="/api/auth/logout">
          {t("nav.sign_out")}
        </a>
      )}
    </nav>
  );
}
