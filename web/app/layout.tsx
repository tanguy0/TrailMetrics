import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "TrailMetrics",
  description:
    "Build your own running-data analysis pages: pick a data source, add the plots you want.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <nav className="topbar">
          <a className="topbar__brand" href="/pages">
            🏔️ TrailMetrics
          </a>
          <div className="topbar__links">
            <a href="/pages">My pages</a>
            <a href="/api/auth/logout">Sign out</a>
          </div>
        </nav>
        {children}
      </body>
    </html>
  );
}
