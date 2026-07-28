/**
 * The app's wording — lookup helpers.
 *
 * There is deliberately no translation table in this repo: every user-facing string
 * lives in `src/translations.py` and is translated server-side, so adding a language
 * there covers the whole product. See that module's docstring.
 *
 * This module is import-safe from client components. Fetching the table is server
 * work and lives in `strings.server.ts` — keeping them apart is what stops
 * `next/headers` being pulled into the browser bundle.
 */

export type Strings = Record<string, string>;

export type Translate = (
  key: string,
  replacements?: Record<string, string | number>,
) => string;

/**
 * Look up a string, falling back to the key.
 *
 * The fallback is deliberate: a missing key renders as `home.profile.title`, which
 * is obvious in review and harmless in production — unlike an empty string, which
 * would silently render as a blank heading.
 */
export function translator(strings: Strings): Translate {
  return (key, replacements) => {
    let text = strings[key] ?? key;
    for (const [name, value] of Object.entries(replacements ?? {})) {
      text = text.replaceAll(`{${name}}`, String(value));
    }
    return text;
  };
}

/** `count` with the right singular/plural key: `ui.pages.panel_count.one|many`. */
export function plural(t: Translate, base: string, count: number): string {
  return t(`${base}.${count === 1 ? "one" : "many"}`, { count });
}
