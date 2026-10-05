"use client";

/**
 * A coach's notifications: the pending coaching requests, on the rail under the
 * athlete switcher (NavRail.md § count).
 *
 * Nothing at all while there is no request — the rail carries no "0". With some,
 * a rail link with their count; it opens the requests in a dialog, each with
 * accept / decline. Accepting adds the athlete to the switcher's roster, which
 * is where a coach opens their athletes from.
 *
 * Re-read on every navigation: the rail stays mounted across pages, and a request
 * that arrives while the coach is browsing should show up on their next click.
 */

import { usePathname } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { Icon } from "@/components/Icon";
import { Modal } from "@/components/Modal";
import { decideCoachingRequest, getCoachBoard } from "@/lib/api";
import { formatDate } from "@/lib/format";
import type { Translate } from "@/lib/strings";
import type { CoachBoard } from "@/lib/types";

export function CoachRequests({ t }: { t: Translate }) {
  const pathname = usePathname();
  const [pending, setPending] = useState<CoachBoard["pending"] | null>(null);
  const [open, setOpen] = useState(false);

  const load = useCallback(() => {
    getCoachBoard()
      .then((board) => setPending(board.pending))
      .catch(() => setPending([]));
  }, []);
  useEffect(load, [load, pathname]);

  const decide = async (id: string, decision: "accept" | "decline") => {
    await decideCoachingRequest(id, decision).catch(() => undefined);
    load();
  };

  if (!pending || (pending.length === 0 && !open)) return null;

  return (
    <div className="shell__switcher">
      <button type="button" className="tm-rail__link shell__requests" onClick={() => setOpen(true)}>
        <Icon name="users" size={17} />
        <span>{t("coaching.requests.nav")}</span>
        {pending.length > 0 && <span className="tm-rail__count">{pending.length}</span>}
      </button>

      {open && (
        <Modal title={t("coaching.requests.title")} onClose={() => setOpen(false)}>
          {pending.length === 0 ? (
            <p className="body-sm muted">{t("coaching.board.none_pending")}</p>
          ) : (
            <ul className="coaching-board__list">
              {pending.map((request) => (
                <li className="coaching-board__item" key={request.id}>
                  <div className="coaching-board__who">
                    <strong>{request.display_name ?? request.email}</strong>
                    <span className="body-sm muted">
                      {request.email}
                      {request.phone ? ` · ${request.phone_e164 ?? request.phone}` : ""}
                      {" · "}
                      {formatDate(request.created_at, "short", t("locale"))}
                    </span>
                    {request.message && (
                      <p className="body-sm coaching-request__message">{request.message}</p>
                    )}
                  </div>
                  <div className="coaching-request__actions">
                    <button
                      type="button"
                      className="tm-btn tm-btn--sm"
                      onClick={() => decide(request.id, "accept")}
                    >
                      {t("coaching.board.accept")}
                    </button>
                    <button
                      type="button"
                      className="tm-btn tm-btn--secondary tm-btn--sm"
                      onClick={() => decide(request.id, "decline")}
                    >
                      {t("coaching.board.decline")}
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Modal>
      )}
    </div>
  );
}
