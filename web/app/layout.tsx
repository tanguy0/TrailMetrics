import type { Metadata } from "next";

import { LanguageSwitcher } from "@/components/LanguageSwitcher";
import { Sidebar } from "@/components/Sidebar";
import { getViewer, lang } from "@/lib/session";
import { loadStrings } from "@/lib/strings.server";

import "./tokens.css";
import "./components.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "TAGG",
  description:
    "Build your own running-data analysis pages: pick a data source, add the plots you want.",
};

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // The rail is always there — brand + wayfinding — even for a visitor with no
  // session yet. `Sidebar` decides from the viewer's tier which links are open.
  const viewer = await getViewer();
  const strings = await loadStrings();

  return (
    <html lang={await lang()}>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        {/* The root layout wraps every route, so this does load app-wide. */}
        {/* eslint-disable-next-line @next/next/no-page-custom-font */}
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap"
        />
      </head>
      <body>
        <div className="shell">
          <Sidebar strings={strings} viewer={viewer} />
          <div className="shell__content">
            <div className="shell__topbar">
              <LanguageSwitcher />
            </div>
            {children}
          </div>
        </div>
      </body>
    </html>
  );
}
