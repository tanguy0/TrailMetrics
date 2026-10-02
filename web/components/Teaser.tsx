/**
 * A locked page for a visitor without an account (design/tagg/visitor.md,
 * components/Teaser.md): the page itself in the background, a card on top that
 * says what is in it and how to get it. No redirect — the visitor keeps the URL,
 * the rail and the context.
 *
 * The background is the page's *empty structure*, drawn with the same classes as
 * the real page (KPI tiles, panels, plot cards, calendar days), so it stays in
 * step with the product rather than going stale like a screenshot would. When a
 * demo athlete exists, "Voir un exemple" will open the real page on its data; until
 * then that button is left out of the DOM, not faked.
 */

import { Icon } from "@/components/Icon";
import { PageHeader } from "@/components/PageHeader";
import { connectStravaHref, loginHref, registerHref } from "@/lib/auth";
import type { Translate } from "@/lib/strings";

export type TeaserPage = "home" | "analysis" | "training";

const PATH: Record<TeaserPage, string> = { home: "/home", analysis: "/pages", training: "/training" };

/**
 * `tier` is what the reader already has: a visitor is invited to create an
 * account, an account without Strava to connect it — never both at once.
 */
export function Teaser({
  page,
  tier = "visitor",
  t,
}: {
  page: TeaserPage;
  tier?: "visitor" | "account";
  t: Translate;
}) {
  const title = t({ home: "nav.home", analysis: "nav.analysis", training: "nav.training" }[page]);
  return (
    <main className="container">
      <PageHeader kicker="TAGG" title={title} />
      <section className="tm-teaser" aria-labelledby="teaser-title">
        <div className="tm-teaser__bg" aria-hidden="true" inert>
          {page === "home" && <HomeStructure t={t} />}
          {page === "analysis" && <AnalysisStructure />}
          {page === "training" && <TrainingStructure />}
        </div>
        <div className="tm-teaser__card">
          <span className="tm-teaser__kicker">
            {t(tier === "visitor" ? "visitor.account.tier" : "visitor.tier")}
          </span>
          <h2 className="tm-teaser__title" id="teaser-title">
            {t(`visitor.teaser.${page}.title`)}
          </h2>
          <ul className="tm-teaser__list">
            {[1, 2, 3].map((n) => (
              <li key={n}>{t(`visitor.teaser.${page}.${n}`)}</li>
            ))}
          </ul>
          <div className="tm-teaser__actions">
            {tier === "visitor" ? (
              <>
                <a className="tm-btn" href={registerHref(PATH[page])}>
                  {t("visitor.register")}
                </a>
                <a className="tm-btn tm-btn--secondary" href={loginHref(PATH[page])}>
                  {t("visitor.login")}
                </a>
              </>
            ) : (
              <a className="tm-btn tm-btn--strava" href={connectStravaHref(PATH[page])}>
                {t("visitor.more.link")}
              </a>
            )}
          </div>
          <p className="tm-teaser__fine">
            {t(tier === "visitor" ? "visitor.account_fine" : "visitor.fine")}
          </p>
        </div>
      </section>
    </main>
  );
}

function EmptyKpi({ label }: { label: string }) {
  return (
    <div className="tm-kpi tm-kpi--flat">
      <span className="tm-kpi__label">{label}</span>
      <span className="tm-kpi__value">
        <span className="tm-kpi__num">—</span>
      </span>
    </div>
  );
}

function EmptyPlot({ span }: { span: number }) {
  return (
    <div className={`tm-plot tm-plot--${span} teaser-plot`}>
      <div className="tm-plot__head">
        <span className="tm-chip">—</span>
      </div>
    </div>
  );
}

function HomeStructure({ t }: { t: Translate }) {
  return (
    <div className="teaser-structure">
      <section className="card-block">
        <h2 className="tm-section section-title">
          <Icon name="run" size={18} />
          <span className="section-title__text">
            <span className="tm-section__kicker">{t("home.kicker.all_time")}</span>
            {t("home.profile.title")}
          </span>
        </h2>
        <div className="kpi-grid kpi-grid--four">
          {["home.profile.activities", "home.profile.total_distance", "home.profile.total_elevation",
            "home.profile.total_time"].map((key) => (
            <EmptyKpi key={key} label={t(key)} />
          ))}
        </div>
      </section>
      <section className="card-block">
        <h2 className="tm-section tm-section--sun section-title">
          <Icon name="award" size={18} />
          <span className="section-title__text">
            <span className="tm-section__kicker">{t("home.kicker.all_time")}</span>
            {t("home.profile.records")}
          </span>
        </h2>
        <div className="kpi-grid kpi-grid--records">
          {["5 km", "10 km", "21,1 km", "42,2 km"].map((label) => (
            <EmptyKpi key={label} label={label} />
          ))}
        </div>
      </section>
    </div>
  );
}

function AnalysisStructure() {
  return (
    <div className="teaser-structure">
      {["01", "02"].map((index) => (
        <section className="tm-panel" key={index}>
          <div className="tm-panel__head">
            <div className="tm-panel__lead">
              <span className="tm-panel__index">{index}</span>
              <div className="tm-panel__meta">
                <span className="tm-chip tm-chip--forest">—</span>
                <span className="tm-chip">—</span>
              </div>
            </div>
          </div>
          <div className="tm-plot-grid">
            <EmptyPlot span={8} />
            <EmptyPlot span={4} />
          </div>
        </section>
      ))}
    </div>
  );
}

function TrainingStructure() {
  return (
    <div className="teaser-structure training-calendar teaser-calendar">
      {[0, 1, 2].map((week) => (
        <div className="training-week" key={week}>
          <div className="training-week__label" />
          <div className="training-week__days">
            {[0, 1, 2, 3, 4, 5, 6].map((day) => (
              <div className={`tm-day training-day${week === 0 && day === 3 ? " tm-day--today" : ""}`} key={day}>
                <div className="tm-day__head">
                  <span>—</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * The one discreet line at the end of an open page (visitor.md § Pages ouvertes):
 * what Strava would add, with a text link — no banner, no modal.
 */
export function StravaMore({ message, next, t }: { message: string; next: string; t: Translate }) {
  return (
    <p className="body-sm visitor-more">
      {message}{" "}
      <a href={connectStravaHref(next)}>{t("visitor.more.link")}</a>
    </p>
  );
}
