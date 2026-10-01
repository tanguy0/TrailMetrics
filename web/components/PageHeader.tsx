/**
 * A page's header (design/tagg/components/PageHeader.md): a mono kicker for the
 * context, the `display-lg` title, an optional muted subtitle, and at most three
 * `sm` actions on the right — one primary at most.
 *
 * `title` is a node so an editable page can pass its name input in its place
 * (`.page-title-input`, which keeps the title's type).
 */

import type { ReactNode } from "react";

export function PageHeader({
  kicker,
  title,
  sub,
  actions,
}: {
  kicker: ReactNode;
  title: ReactNode;
  sub?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="tm-page-header page-header">
      <div className="page-header__lead">
        <div className="tm-page-header__kicker">{kicker}</div>
        <h1 className="tm-page-header__title">{title}</h1>
        {sub && <p className="tm-page-header__sub">{sub}</p>}
      </div>
      {actions && <div className="tm-page-header__actions">{actions}</div>}
    </header>
  );
}
