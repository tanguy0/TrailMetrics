/** The page a reset link opens: choose a new password. The token is checked on submit. */

import type { Metadata } from "next";

import { ResetForm } from "@/components/ResetForm";
import { loadStrings } from "@/lib/strings.server";

export const metadata: Metadata = {
  title: "TAGG",
  // The token is in the URL: keep it out of any Referer this page could send.
  referrer: "no-referrer",
};

export default async function ResetTokenPage({ params }: { params: Promise<{ token: string }> }) {
  const { token } = await params;
  return (
    <main className="container auth-page">
      <ResetForm token={token} contact={null} strings={await loadStrings()} />
    </main>
  );
}
