"use client";

/**
 * The Tools tab's sub-tabs (design/tagg/access.md § Navigation): Race Planning ·
 * Level Assessment · GAP Profile · Durability Profile. A `tm-segment` of links, one URL
 * per tool. The two that read Strava carry the lock when the viewer has no
 * Strava yet — still clickable, to the tool's teaser (rule v1.1).
 */

import { usePathname } from "next/navigation";

import { Icon } from "@/components/Icon";
import { translator, type Strings } from "@/lib/strings";

const TABS = [
  { href: "/tools/race-planning", label: "tools.race_planning", strava: false },
  { href: "/tools/level", label: "tools.level", strava: false },
  { href: "/tools/gap", label: "tools.gap_profile", strava: true },
  { href: "/tools/durability", label: "tools.durability", strava: true },
];

export function ToolsTabs({ strings, hasStrava }: { strings: Strings; hasStrava: boolean }) {
  const t = translator(strings);
  const pathname = usePathname() ?? "";
  return (
    <nav className="container tools-tabs" aria-label={t("nav.tools")}>
      <div className="tm-segment" role="tablist">
        {TABS.map((tab) => {
          const active = pathname === tab.href || pathname.startsWith(`${tab.href}/`);
          return (
            <a
              key={tab.href}
              className="tm-segment__item"
              role="tab"
              aria-selected={active}
              href={tab.href}
            >
              {t(tab.label)}
              {tab.strava && !hasStrava && <Icon name="lock" size={12} className="tm-lock" />}
            </a>
          );
        })}
      </div>
    </nav>
  );
}
