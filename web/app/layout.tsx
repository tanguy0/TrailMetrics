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
  // The rail is navigation between an athlete's own screens, so it only exists
  // once there is a session — the sign-in page has nowhere to navigate to.
  const signedIn = Boolean(await readSession());
  const strings = signedIn ? await loadStrings() : {};

  return (
    <html lang={await lang()}>
      <body>
        {signedIn ? (
          <div className="shell">
            <Sidebar strings={strings} />
            <div className="shell__content">
              <div className="shell__topbar">
                <LanguageSwitcher />
              </div>
              {children}
            </div>
          </div>
        ) : (
          children
        )}
      </body>
    </html>
  );
}
