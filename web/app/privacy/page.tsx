import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Privacy Policy — TrailMetrics",
};

export default function PrivacyPage() {
  return (
    <main className="container container--narrow">
      <h1>Privacy Policy</h1>
      <p className="muted">Last updated: 6 August 2026</p>

      <p>
        TrailMetrics is a personal project built and operated by Tanguy Blervacque. This
        page explains what data the app collects, why, and how to get it deleted.
      </p>

      <h2>Who this is</h2>
      <p>
        Tanguy Blervacque is the sole operator of TrailMetrics and the data controller
        for anything it stores. Contact:{" "}
        <a href="mailto:tanguy.blervacque@gmail.com">tanguy.blervacque@gmail.com</a>.
      </p>

      <h2>What is collected</h2>
      <p>Signing in with Strava and using the app stores:</p>
      <ul>
        <li>Your Strava name, profile picture URL, and athlete ID.</li>
        <li>
          Your Strava access and refresh tokens, encrypted at rest — used only to fetch
          your own activities from Strava's API.
        </li>
        <li>Your activities: dates, distance, elevation, pace, heart rate, power, GPS-derived route data, and per-second streams for the ones that have them.</li>
        <li>
          Data you type in yourself: email address, weight, birthdate, height, heart-rate
          zones, VMA pace, training-diary entries, and any pages or images you create in
          the app.
        </li>
      </ul>
      <p>
        Strava does not hand over an email address, which is why the app asks for one
        directly after your first sign-in.
      </p>

      <h2>Why</h2>
      <p>
        Every piece of data above exists to run the analysis you asked for — plotting
        your own training, fitting models on your own runs — and for nothing else.
      </p>

      <h2>Coaching access</h2>
      <p>
        Tanguy also coaches some of the people using this app. When he does, his
        account can browse your account to see your training data — activities,
        plots, and your training diary — and can add or edit entries in your
        training calendar (planned workouts and goals) to help plan your training.
      </p>
      <p>
        He cannot connect, reconnect, or trigger an import of your Strava data —
        only you can do that, from your own account.
      </p>

      <h2>Who else sees it</h2>
      <p>
        Nobody. There is no advertising, no analytics or tracking script, and your data
        is never sold or shared with third parties, beyond the infrastructure providers
        that host the app and necessarily process data on its behalf:
      </p>
      <ul>
        <li>Strava — the source of your activity data (via OAuth you grant directly).</li>
        <li>Supabase — hosts the database and file storage.</li>
        <li>Vercel and Railway — host the web app and the compute service.</li>
      </ul>

      <h2>How long it's kept</h2>
      <p>
        For as long as your account exists, so the app can keep showing your training
        history. Nothing is retained beyond that once an account is deleted.
      </p>

      <h2>Your rights</h2>
      <p>
        You can ask for a copy of everything stored about you, or for your account and
        all associated data to be permanently deleted, at any time, by emailing{" "}
        <a href="mailto:tanguy.blervacque@gmail.com">tanguy.blervacque@gmail.com</a>.
        Deletion removes your athlete record and everything linked to it — activities,
        credentials, pages, and diary entries — and cannot be undone.
      </p>

      <h2>Security</h2>
      <p>
        Strava tokens are encrypted before being stored. Sessions are held in an
        httpOnly cookie your browser's JavaScript cannot read. The app is not perfect —
        no software is — but it is built and reviewed with your data in mind.
      </p>
    </main>
  );
}
