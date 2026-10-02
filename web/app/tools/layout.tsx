/** Every tool under one tab, with the sub-tabs above it. */

import { ToolsTabs } from "@/components/ToolsTabs";
import { getViewer } from "@/lib/session";
import { loadStrings } from "@/lib/strings.server";

export default async function ToolsLayout({ children }: { children: React.ReactNode }) {
  const viewer = await getViewer();
  return (
    <>
      <ToolsTabs strings={await loadStrings()} hasStrava={viewer?.tier === "strava"} />
      {children}
    </>
  );
}
