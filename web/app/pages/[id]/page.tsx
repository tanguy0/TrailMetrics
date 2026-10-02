import { redirect } from "next/navigation";

import { PageWorkspace } from "@/components/PageWorkspace";
import { getViewer } from "@/lib/session";
import { loadStrings } from "@/lib/strings.server";

export default async function EditPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  // Analyses read Strava data; without it, the tab's teaser says so.
  if ((await getViewer())?.tier !== "strava") redirect("/pages");
  const { id } = await params;
  // Read server-side and passed in, like every other screen: fetching in the browser
  // would flash untranslated keys on first paint.
  return <PageWorkspace pageId={id} strings={await loadStrings()} />;
}
