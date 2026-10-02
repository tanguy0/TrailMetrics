/** Tools → Level Assessment. Open to everyone; an account keeps the result. */

import type { Metadata } from "next";

import { LevelScreen } from "@/components/LevelScreen";
import { getViewer } from "@/lib/session";
import { loadStrings } from "@/lib/strings.server";

export const metadata: Metadata = { title: "Évaluation du niveau — TAGG" };

export default async function LevelPage() {
  const viewer = await getViewer();
  return <LevelScreen strings={await loadStrings()} signedIn={Boolean(viewer)} />;
}
