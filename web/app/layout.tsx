import type { Metadata } from "next";

import { LanguageSwitcher } from "@/components/LanguageSwitcher";
import { Sidebar } from "@/components/Sidebar";
import { lang, readSession } from "@/lib/session";
import { loadStrings } from "@/lib/strings.server";

import "./globals.css";

export const metadata: Metadata = {
  title: "TrailMetrics",
  description:
    "Build your own running-data analysis pages: pick a data source, add the plots you want.",
};

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // The rail is always there — brand + wayfinding — even for a visitor with no
  // session yet. `Sidebar` is the one that decides which of its links actually
  // go anywhere.
  const signedIn = Boolean(await readSession());
  const strings = await loadStrings();

  return (
    <html lang={await lang()}>
      <body>
        <div className="shell">
          <Sidebar strings={strings} authenticated={signedIn} />
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
