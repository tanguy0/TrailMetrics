/**
 * Landing page: connect Strava, or go straight through if already signed in.
 *
 * A server component so the session cookie decides before anything renders — no
 * flash of a sign-in screen for a signed-in user.
 */

import { redirect } from "next/navigation";

import { readSession } from "@/lib/session";

export default async function Home({
  searchParams,
}: {
  searchParams: Promise<{ error?: string }>;
}) {
  if (await readSession()) redirect("/home");
  const { error } = await searchParams;

  return (
    <main className="container container--narrow">
      <h1>Analyse your running, your way</h1>
      <p className="lede">
        TrailMetrics is a data-science workbench for running. You build the pages: choose
        a data source — specific runs, a date range, or several periods to compare — then
        add the plots you want over it.
      </p>

      {error && <p className="note note--error">{error}</p>}

      <a className="button button--strava" href="/api/auth/strava/start">
        Connect with Strava
      </a>

      <section className="feature-list">
        <div>
          <h3>Panels, not fixed dashboards</h3>
          <p>
            A panel has one data source and as many plots as you like. Compare training
            blocks by defining them as time windows.
          </p>
        </div>
        <div>
          <h3>Every metric, every chart form</h3>
          <p>
            Distance, elevation, gradient, pace, GAP, power-to-heart-rate, best efforts —
            as trends, distributions, scatter plots or tables.
          </p>
        </div>
        <div>
          <h3>Models on your own data</h3>
          <p>
            Fit a personalized gradient-adjusted-pace curve on any selection and compare
            it against reference curves.
          </p>
        </div>
        <div>
          <h3>Examples you can take apart</h3>
          <p>
            Three ready-made pages ship with the app. Duplicate one and edit it — they
            are built from the same panels you get.
          </p>
        </div>
      </section>

      <p className="muted">
        TrailMetrics reads your activities from Strava so it can analyse them. Your Strava
        tokens are encrypted and never leave the server.
      </p>

      <p className="muted">
        <a href="/privacy">Privacy Policy</a> · <a href="/terms">Terms of Service</a>
      </p>
    </main>
  );
}
