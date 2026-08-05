import { PageWorkspace } from "@/components/PageWorkspace";
import { loadStrings } from "@/lib/strings.server";

export default async function EditPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  // Read server-side and passed in, like every other screen: fetching in the browser
  // would flash untranslated keys on first paint.
  return <PageWorkspace pageId={id} strings={await loadStrings()} />;
}
