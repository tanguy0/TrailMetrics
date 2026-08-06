"use client";

/**
 * A plain overlay + centered panel. There is no component library in this repo
 * (see `globals.css`'s module docstring), so this is the one hand-rolled dialog
 * shared by every popup the app needs — today the Training calendar's item editor
 * and its session-detail view.
 */

import { useEffect } from "react";

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
        className={`modal-panel${wide ? " modal-panel--wide" : ""}`}
        role="dialog"
        aria-modal="true"
        aria-label={title}
      >
        <div className="modal-panel__header">
          <h3 className="modal-panel__title">{title}</h3>
          <button
            type="button"
            className="modal-panel__close"
            onClick={onClose}
            aria-label="Close"
          >
            ×
          </button>
        </div>
        <div className="modal-panel__body">{children}</div>
      </div>
    </div>
  );
}
