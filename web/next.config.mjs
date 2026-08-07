// Content-Security-Policy is set in middleware.ts, not here: script-src needs a
// fresh nonce on every request (Next's App Router injects its own inline
// hydration scripts on every page), and next.config.mjs's `headers()` has no way
// to vary its value per request. The rest of these don't need that, so they stay
// static and live here.
const SECURITY_HEADERS = [
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Referrer-Policy", value: "no-referrer" },
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
  // Harmless to send over plain HTTP in local dev — browsers only act on it once
  // seen over a real HTTPS connection, which is all that's ever public.
  { key: "Strict-Transport-Security", value: "max-age=63072000; includeSubDomains" },
];

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // plotly.js-dist-min is a large prebuilt bundle; it is only ever imported from
  // client components behind a dynamic import, so it never enters the server build.
  experimental: {
    optimizePackageImports: [],
  },
  async headers() {
    return [{ source: "/:path*", headers: SECURITY_HEADERS }];
  },
};

export default nextConfig;
