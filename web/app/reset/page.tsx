/**
 * Ask for a reset link. When the API has no mail provider configured, the page
 * says whom to write to instead of offering a form that would send nothing.
 */

import type { Metadata } from "next";

import { ResetForm } from "@/components/ResetForm";
import { apiBaseUrl } from "@/lib/session";
import { loadStrings } from "@/lib/strings.server";

export const metadata: Metadata = { title: "TAGG" };

async function contactIfNoMail(): Promise<string | null> {
  try {
    const response = await fetch(`${apiBaseUrl()}/auth/reset/options`, { cache: "no-store" });
    if (!response.ok) return null;
    const body = (await response.json()) as { can_send: boolean; contact: string };
    return body.can_send ? null : body.contact;
  } catch {
    return null;
  }
}

export default async function ResetPage() {
  return (
    <main className="container auth-page">
      <ResetForm token={null} contact={await contactIfNoMail()} strings={await loadStrings()} />
    </main>
  );
}
