"use client";

/**
 * The Coaching tab (design/specs/coaching.md § Écrans, design/tagg/access.md).
 *
 *  - Not coached: the page that has to make people want it — styled as an offer
 *    (`tm-offer`: strong title, three points, proof figures from the database,
 *    the diary's empty structure as a preview), then the request form in a card.
 *    States: to fill in, sent on … (edit, withdraw), declined.
 *  - Coached: the diary (TrainingScreen) — it needs Strava to fill in.
 *  - Coach: like a coached athlete — their own diary. Requests are answered
 *    from the rail (`CoachRequests`), athletes opened from its switcher. While
 *    viewing an athlete, their diary, with "My athletes" to come back.
 *
 * One `bg-hero` on the page, the offer's: nothing else here is a hero.
 */

import { useCallback, useEffect, useState, type FormEvent } from "react";

import { Callout } from "@/components/Callout";
import { TrainingScreen } from "@/components/TrainingScreen";
import {
  getAthlete,
  getCoachingState,
  putCoachingRequest,
  withdrawCoachingRequest,
} from "@/lib/api";
import { connectStravaHref } from "@/lib/auth";
import { formatDate, formatNumber } from "@/lib/format";
import { translator, type Strings, type Translate } from "@/lib/strings";
import type { CoachingState } from "@/lib/types";

async function viewAs(athleteId: number | null) {
  await fetch("/api/view-as", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ athleteId }),
  });
  window.location.assign("/coaching");
}

export function CoachingScreen({
  strings,
  isCoach,
  hasStrava,
}: {
  strings: Strings;
  isCoach: boolean;
  hasStrava: boolean;
}) {
  const t = translator(strings);
  const [state, setState] = useState<CoachingState | null>(null);
  const [viewing, setViewing] = useState<boolean | null>(isCoach ? null : false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setState(await getCoachingState());
    } catch (caught) {
      setError((caught as Error).message);
    }
  }, []);

  useEffect(() => {
    load();
    if (isCoach) {
      getAthlete().then((me) => setViewing(me.viewing_as)).catch(() => setViewing(false));
    }
  }, [load, isCoach]);

  if (error) {
    return (
      <main className="container">
        <Callout tone="terra">{error}</Callout>
      </main>
    );
  }
  if (!state || viewing === null) {
    return (
      <main className="container">
        <p className="muted">{t("common.loading")}</p>
      </main>
    );
  }

  if (isCoach && viewing) {
    return (
      <>
        <div className="container coaching-bar">
          <button type="button" className="tm-btn tm-btn--secondary tm-btn--sm" onClick={() => viewAs(null)}>
            {t("coaching.my_athletes")}
          </button>
        </div>
        <TrainingScreen strings={strings} />
      </>
    );
  }

  if (state.coached || isCoach) {
    if (hasStrava) return <TrainingScreen strings={strings} />;
    return (
      <main className="container coaching">
        <Callout>
          {t("coaching.needs_strava")}{" "}
          <a href={connectStravaHref("/coaching")}>{t("home.strava.connect")}</a>
        </Callout>
      </main>
    );
  }

  return (
    <main className="container coaching">
      <Offer state={state} t={t} />
      <RequestCard state={state} onChanged={load} t={t} />
    </main>
  );
}

// --- The offer ----------------------------------------------------------------

const PREVIEW_DAYS = ["L", "M", "M", "J", "V", "S", "D"];

function Offer({ state, t }: { state: CoachingState; t: Translate }) {
  return (
    <section className="tm-offer">
      <div className="coaching-offer__copy">
        <span className="tm-hero__kicker">{t("coaching.offer.kicker")}</span>
        <h1 className="tm-offer__title">{t("coaching.offer.title")}</h1>
        <ul className="tm-offer__list">
          {[1, 2, 3].map((n) => <li key={n}>{t(`coaching.offer.${n}`)}</li>)}
        </ul>
        {state.proof && (
          <div className="tm-offer__proof">
            <div>
              <div className="v is-key">{formatNumber(state.proof.coached_count, 0)}</div>
              <div className="l">{t("coaching.offer.proof")}</div>
            </div>
          </div>
        )}
      </div>
      {/* The diary's real structure, empty: what the athlete would get. */}
      <div className="tm-offer__preview" aria-hidden="true">
        <span className="coaching-preview__title">{t("coaching.offer.preview")}</span>
        {PREVIEW_DAYS.map((day, index) => (
          <div className="coaching-preview__row" key={index}>
            <span className="coaching-preview__day">{day}</span>
            <span className="coaching-preview__slot">{index === 0 ? t("coaching.offer.planned") : ""}</span>
            <span className="coaching-preview__slot">{index === 0 ? t("coaching.offer.done") : ""}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

// --- The request ----------------------------------------------------------------

function RequestCard({
  state,
  onChanged,
  t,
}: {
  state: CoachingState;
  onChanged: () => void;
  t: Translate;
}) {
  const request = state.request;
  const pending = request?.status === "pending";
  const declined = request?.status === "declined";
  const [editing, setEditing] = useState(!pending && !declined);
  const [message, setMessage] = useState(pending ? request.message : "");
  const [contact, setContact] = useState<"email" | "phone">(pending ? request.contact : "email");
  const [phone, setPhone] = useState(pending ? request.phone ?? "" : "");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setEditing(!(request?.status === "pending") && !(request?.status === "declined"));
  }, [request?.status]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await putCoachingRequest({ message, contact, phone: phone.trim() || null });
      setEditing(false);
      onChanged();
    } catch (caught) {
      setError((caught as Error).message);
    }
    setBusy(false);
  }

  if (declined && !editing) {
    return (
      <section className="card-block coaching-request">
        <p className="body">{t("coaching.state.declined")}</p>
        {state.can_request_again_at ? (
          <p className="body-sm muted">
            {t("coaching.state.again_on", { date: formatDate(state.can_request_again_at, "short", t("locale")) })}
          </p>
        ) : (
          <div>
            <button type="button" className="tm-btn" onClick={() => setEditing(true)}>
              {t("coaching.form.title")}
            </button>
          </div>
        )}
      </section>
    );
  }

  if (pending && !editing) {
    return (
      <section className="card-block coaching-request">
        <p className="body">
          {t("coaching.state.sent", { date: formatDate(request.created_at, "short", t("locale")) })}
        </p>
        {request.message && <p className="body-sm muted coaching-request__message">{request.message}</p>}
        <div className="coaching-request__actions">
          <button type="button" className="tm-btn tm-btn--secondary tm-btn--sm" onClick={() => setEditing(true)}>
            {t("coaching.state.edit")}
          </button>
          <button
            type="button"
            className="tm-btn tm-btn--ghost tm-btn--sm"
            onClick={async () => {
              await withdrawCoachingRequest().catch(() => undefined);
              onChanged();
            }}
          >
            {t("coaching.state.withdraw")}
          </button>
        </div>
      </section>
    );
  }

  return (
    <form className="card-block coaching-request" onSubmit={submit} noValidate>
      <h2 className="tm-section section-title">
        <span className="section-title__text">{t("coaching.form.title")}</span>
      </h2>
      {error && <Callout tone="terra">{error}</Callout>}
      <label className="tm-field">
        {t("coaching.form.message")}
        <textarea
          className="tm-textarea"
          rows={5}
          maxLength={4000}
          value={message}
          onChange={(event) => setMessage(event.target.value)}
        />
      </label>
      <div className="tm-field">
        {t("coaching.form.contact")}
        {/* A tablist, not radios: tm-segment shows its choice through aria-selected. */}
        <div className="tm-segment coaching-request__contact" role="tablist">
          {(["email", "phone"] as const).map((key) => (
            <button
              key={key}
              type="button"
              role="tab"
              className="tm-segment__item"
              aria-selected={contact === key}
              onClick={() => setContact(key)}
            >
              {t(key === "email" ? "coaching.form.by_email" : "coaching.form.by_phone")}
            </button>
          ))}
        </div>
      </div>
      {contact === "email" ? (
        <label className="tm-field">
          {t("auth.email")}
          <input className="tm-input" value={state.email} readOnly />
        </label>
      ) : (
        <label className="tm-field">
          {t("coaching.form.phone")}
          <input
            className="tm-input"
            type="tel"
            autoComplete="tel"
            value={phone}
            onChange={(event) => setPhone(event.target.value)}
          />
        </label>
      )}
      <div>
        <button type="submit" className="tm-btn" disabled={busy}>
          {t(pending ? "coaching.form.save" : "coaching.form.send")}
        </button>
      </div>
    </form>
  );
}
