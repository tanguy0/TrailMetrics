/**
 * Where "Connect with Strava" goes: the OAuth start route, which sends the athlete
 * back to `next` once connected — a visitor lands back on the page they were on.
 */
export function signInHref(next?: string): string {
  const start = "/api/auth/strava/start";
  return next ? `${start}?next=${encodeURIComponent(next)}` : start;
}
