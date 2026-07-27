/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // plotly.js-dist-min is a large prebuilt bundle; it is only ever imported from
  // client components behind a dynamic import, so it never enters the server build.
  experimental: {
    optimizePackageImports: [],
  },
};

export default nextConfig;
