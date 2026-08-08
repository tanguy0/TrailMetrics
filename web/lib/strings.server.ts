/**
 * Server-side fetch of the UI string table.
 *
 * Separate from `strings.ts` because this reaches for the API URL and the request
 * language, which are server concerns. Strings are read here and passed into client
 * components as props: fetching them in the browser would flash untranslated keys on
 * first paint, and the payload is small enough that inlining it costs nothing.
 */

import { apiBaseUrl, lang } from "./session";
import type { Strings } from "./strings";
import type { UiStrings } from "./types";

/**
 * Cached for an hour: the table only changes when the Python source does, and a
 * stale label is a far smaller problem than a round-trip on every navigation.
 *
 * Falls back to an empty table so a cold API degrades to visible keys rather than a
 * crash — the shell still renders, and the data endpoints report the real error.
 */
export async function loadStrings(): Promise<Strings> {
  try {
    const response = await fetch(
      `${apiBaseUrl()}/ui-strings?lang=${encodeURIComponent(await lang())}`,
      { next: { revalidate: 3600 } },
    );
    if (!response.ok) return {};
    return ((await response.json()) as UiStrings).strings ?? {};
  } catch {
    return {};
  }
}
