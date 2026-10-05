"use client";

/**
 * The signed-in Race plan tab: every saved plan, newest first, then the button
 * to start another — the same shape as the Analysis tab's list of pages.
 *
 * A plan marked as an objective takes the diary's goal colours (design/tagg
 * SessionCard `goal`): main in terra, secondary in sun.
 */

import Link from "next/link";
import { useEffect, useState } from "react";

import { Callout } from "@/components/Callout";
import { ElevationThumb } from "@/components/ElevationThumb";
import { PageHeader } from "@/components/PageHeader";
import { RouteMap } from "@/components/RouteMap";
import { listRacePlans } from "@/lib/api";
import { formatDate, formatHms, formatNumber } from "@/lib/format";
import { translator, type Strings, type Translate } from "@/lib/strings";
import type { SavedRacePlan } from "@/lib/types";

/** "Modifié il y a 5 min" / "Modifié hier", but "Modifié le 3 oct." past a week. */
function updatedLabel(at: string, t: Translate): string {
  const relative = formatDate(at, "relative", t("locale"));
  const calendar = formatDate(at, "short", t("locale"));
  return t(relative === calendar ? "race_plan.updated_on" : "race_plan.updated", { date: relative });
}

export function RacePlanList({ strings }: { strings: Strings }) {
  const t = translator(strings);
  const [plans, setPlans] = useState<SavedRacePlan[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listRacePlans().then(setPlans).catch((e: Error) => setError(e.message));
  }, []);

  return (
    <main className="container">
      <PageHeader kicker={t("tools.race_planning")} title={t("race_plan.title")} sub={t("race_plan.intro")} />

      {error && <Callout tone="terra">{error}</Callout>}
      {plans == null && !error ? (
        <p className="muted">{t("common.loading")}</p>
      ) : plans && plans.length === 0 ? (
        <p className="muted">{t("race_plan.empty")}</p>
      ) : (
        <div className="race-plan-list">
          {(plans ?? []).map((plan) => (
            <a
              className={`card race-plan-card${plan.importance ? ` race-plan-card--${plan.importance}` : ""}`}
              key={plan.id}
              href={`/tools/race-planning/${plan.id}`}
            >
              <span className="race-plan-card__summary">
              <span className="card__title">{plan.title || t("race_plan.untitled")}</span>
              {(plan.event_date || plan.importance) && (
                <span className="race-plan-card__event">
                  {[
                    plan.event_date && formatDate(plan.event_date, "long", t("locale")),
                    plan.importance && t(`race_plan.importance.${plan.importance}`),
                  ]
                    .filter(Boolean)
                    .join(" · ")}
                </span>
              )}
              <span className="card__meta">
                {[
                  plan.distance_m != null && `${formatNumber(plan.distance_m / 1000, 1)} km`,
                  plan.elevation_gain_m != null &&
                    `D+ ${formatNumber(plan.elevation_gain_m, 0)} m`,
                  formatHms(plan.params.target_time_s, { exact: true }),
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </span>
              {plan.updated_at && (
                <span className="card__description">
                  {updatedLabel(plan.updated_at, t)}
                </span>
              )}
              </span>
              {plan.preview && (
                <>
                  <span className="race-plan-card__map">
                    <RouteMap points={plan.preview.route} height={128} interactive={false} />
                  </span>
                  <span className="race-plan-card__profile">
                    <ElevationThumb profile={plan.preview.profile} />
                  </span>
                </>
              )}
            </a>
          ))}
        </div>
      )}

      <Link className="new-page" href="/tools/race-planning/new">
        <span className="new-page__plus" aria-hidden="true">+</span>
        <span className="new-page__text">
          <span className="new-page__label">{t("race_plan.new.button")}</span>
          <span className="new-page__hint">{t("race_plan.new.hint")}</span>
        </span>
      </Link>
    </main>
  );
}
