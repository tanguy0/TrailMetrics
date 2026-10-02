/**
 * A note that has to be read (`tm-callout`, design/tagg/contrast.md § Au niveau
 * du texte): sun for information, terra for an alert or an error, forest for a
 * piece of advice. At most one per card — it replaces the old grey `.note`.
 */

import type { ReactNode } from "react";

import { Icon } from "@/components/Icon";

export function Callout({
  tone = "sun",
  children,
}: {
  tone?: "sun" | "terra" | "forest";
  children: ReactNode;
}) {
  return (
    <div className={`tm-callout${tone === "sun" ? "" : ` tm-callout--${tone}`}`} role={tone === "terra" ? "alert" : undefined}>
      <Icon name={tone === "terra" ? "alert" : "info"} />
      <span>{children}</span>
    </div>
  );
}
