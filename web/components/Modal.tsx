"use client";

/**
 * A plain overlay + centered panel. There is no component library in this repo
 * (see `globals.css`'s module docstring), so this is the one hand-rolled dialog
 * shared by every popup the app needs — today the Training calendar's item editor
 * and its session-detail view.
 */

import { useEffect } from "react";

import { Icon } from "@/components/Icon";

export function Modal({
  title,
  onClose,
  wide,
  children,
}: {
  title: string;
  onClose: () => void;
  /** Full-width panel — for content that wants the room, like a session's map
   * and charts, rather than the default form-sized dialog. */
  wide?: boolean;
  children: React.ReactNode;
}) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  return (
    <div
      className="modal-overlay"
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        className={`tm-modal${wide ? " tm-modal--wide" : ""}`}
        role="dialog"
        aria-modal="true"
        aria-label={title}
      >
        <div className="tm-modal__head">
          <h3 className="tm-modal__title">{title}</h3>
          <button
            type="button"
            className="tm-btn tm-btn--ghost tm-btn--icon"
            onClick={onClose}
            aria-label="Close"
          >
            <Icon name="x" />
          </button>
        </div>
        <div className="tm-modal__body">{children}</div>
      </div>
    </div>
  );
}
