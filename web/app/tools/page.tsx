import { redirect } from "next/navigation";

/** The Tools tab opens on its first tool. */
export default function ToolsPage() {
  redirect("/tools/race-planning");
}
