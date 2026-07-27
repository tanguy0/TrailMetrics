import { PageWorkspace } from "@/components/PageWorkspace";

export default async function EditPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <PageWorkspace pageId={id} />;
}
