"use client";

/**
 * The signed-in Race plan tab: every saved plan, newest first, then the button
 * to start another — the same shape as the Analysis tab's list of pages.
 */

import Link from "next/link";
import { useEffect, useState } from "react";

import { PageHeader } from "@/components/PageHeader";
import { listRacePlans } from "@/lib/api";
import { formatDate, formatHms, formatNumber } from "@/lib/format";
import { translator, type Strings } from "@/lib/strings";
import type { SavedRacePlan } from "@/lib/types";

export function RacePlanList({ strings }: { strings: Strings }) {
  const t = translator(strings);
  const [plans, setPlans] = useState<SavedRacePlan[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listRacePlans().then(setPlans).catch((e: Error) => setError(e.message));
  }, []);

  return (
    <main className="container">
      <PageHeader kicker={t("nav.race_plan")} title={t("race_plan.title")} sub={t("race_plan.intro")} />

      {error && <p className="note note--error">{error}</p>}
      {plans == null && !error ? (
        <p className="muted">{t("common.loading")}</p>
      ) : plans && plans.length === 0 ? (
        <p className="muted">{t("race_plan.empty")}</p>
      ) : (
        <div className="card-grid">
          {(plans ?? []).map((plan) => (
            <a className="card" key={plan.id} href={`/race-plan/${plan.id}`}>
              <span className="card__title">{plan.title || t("race_plan.untitled")}</span>
              <span className="card__meta">
                {[
                  plan.distance_m != null && `${formatNumber(plan.distance_m / 1000, 1)} km`,
                  plan.elevation_gain_m != null &&
                    `D+ ${formatNumber(plan.elevation_gain_m, 0)} m`,
                  formatHms(plan.params.target_time_s),
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </span>
              {plan.updated_at && (
                <span className="card__description">
                  {t("race_plan.updated", { date: formatDate(plan.updated_at) })}
                </span>
              )}
            </a>
          ))}
        </div>
      )}

      <Link className="new-page" href="/race-plan/new">
        <span className="new-page__plus" aria-hidden="true">+</span>
        <span className="new-page__text">
          <span className="new-page__label">{t("race_plan.new.button")}</span>
          <span className="new-page__hint">{t("race_plan.new.hint")}</span>
        </span>
      </Link>
    </main>
  );
}
