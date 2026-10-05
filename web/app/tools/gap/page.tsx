/** Tools → Profil GAP. Reads Strava data: without it, the tool's teaser. */

import type { Metadata } from "next";

import { Teaser } from "@/components/Teaser";
import { GapScreen } from "@/components/ToolScreens";
import { getViewer } from "@/lib/session";
import { translator } from "@/lib/strings";
import { loadStrings } from "@/lib/strings.server";

export const metadata: Metadata = { title: "Profil GAP — TAGG" };

export default async function Page() {
  const strings = await loadStrings();
  const viewer = await getViewer();
  if (viewer?.tier !== "strava") {
    return <Teaser page="gap" tier={viewer ? "account" : "visitor"} t={translator(strings)} />;
  }
  return <GapScreen strings={strings} />;
}
