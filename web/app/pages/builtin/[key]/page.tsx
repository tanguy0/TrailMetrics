import { PageWorkspace } from "@/components/PageWorkspace";

/** A built-in example: same workspace, read-only, with a Duplicate action. */
export default async function BuiltinPage({
  params,
}: {
  params: Promise<{ key: string }>;
}) {
  const { key } = await params;
  return <PageWorkspace builtinKey={key} />;
}
