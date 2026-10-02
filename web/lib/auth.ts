/**
 * Where the account and Strava links go. Each carries `next` so the reader lands
 * back on the page they were on.
 *
 * "Connect Strava" works from anywhere: its start route sends a visitor through
 * account creation first, then straight on to Strava (design/specs/auth.md —
 * Strava attaches to an account, it no longer signs anyone in).
 */

function withNext(path: string, next?: string): string {
  return next ? `${path}?next=${encodeURIComponent(next)}` : path;
}

export const connectStravaHref = (next?: string) => withNext("/api/auth/strava/start", next);
export const registerHref = (next?: string) => withNext("/register", next);
export const loginHref = (next?: string) => withNext("/login", next);
