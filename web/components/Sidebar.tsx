"use client";

/**
 * The signed-in navigation rail.
 *
 * A client component only because the active item depends on the current path;
 * the labels arrive already translated from the server, so nothing is fetched here.
 */

import { usePathname } from "next/navigation";

import { translator, type Strings } from "@/lib/strings";

export function Sidebar({ strings }: { strings: Strings }) {
  const t = translator(strings);
  const pathname = usePathname() ?? "";

  const items = [
    { href: "/home", label: t("nav.home"), icon: "🏠" },
    { href: "/pages", label: t("nav.analysis"), icon: "📊" },
    { href: "/training", label: t("nav.training"), icon: "📅" },
  ];

  return (
    <nav className="sidebar" aria-label={t("nav.analysis")}>
      <a className="sidebar__brand" href="/home">
        <span className="sidebar__brand-mark">🏔️</span>
        <span className="sidebar__brand-name">TrailMetrics</span>
      </a>

      <ul className="sidebar__nav">
        {items.map((item) => {
          // `startsWith` so a page being edited (/pages/abc) keeps its tab lit.
          const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
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

      <a className="sidebar__signout" href="/api/auth/logout">
        {t("nav.sign_out")}
      </a>
    </nav>
  );
}
