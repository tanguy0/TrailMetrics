import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Terms of Service — TrailMetrics",
};

export default function TermsPage() {
  return (
    <main className="container container--narrow">
      <h1>Terms of Service</h1>
      <p className="muted">Last updated: 6 August 2026</p>

      <p>
        TrailMetrics is a personal project, run by Tanguy Blervacque, made available to
        a small group of people he knows directly. By signing in you agree to the
        following.
      </p>

      <h2>What this is</h2>
      <p>
        A running-data analysis tool built on top of your own Strava data. It is offered
        as-is, free of charge, without any guarantee of uptime, correctness, or
        continued availability. It could change or disappear without notice — this is a
        side project, not a product with a support commitment.
      </p>

      <h2>Your account</h2>
      <p>
        You connect via Strava OAuth and grant the app read access to your activities.
        You are responsible for the accuracy of anything you type in yourself (weight,
        heart-rate zones, and similar). You can revoke access from Strava's own
        settings at any time, and can ask for your account and data to be deleted — see
        the <a href="/privacy">Privacy Policy</a> for how.
      </p>

      <h2>No warranty</h2>
      <p>
        The analysis, charts, and model fits are for your own informational and training
        use. They are not medical or professional coaching advice, and nothing here is
        guaranteed to be accurate. Use your own judgment, especially around health
        metrics like heart rate and training load.
      </p>

      <h2>Fair use</h2>
      <p>
        Don't use the app to access data that isn't yours, attempt to disrupt it, or
        automate requests against it outside of normal use.
      </p>

      <h2>Changes</h2>
      <p>
        These terms may change as the project evolves. Continuing to use the app after a
        change means you accept the updated terms.
      </p>

      <h2>Contact</h2>
      <p>
        Questions go to{" "}
        <a href="mailto:tanguy.blervacque@gmail.com">tanguy.blervacque@gmail.com</a>.
      </p>
    </main>
  );
}
