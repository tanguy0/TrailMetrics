"use client";

/**
 * The UI string table for components too deep to be handed `strings` — the chart
 * renderer sits under a dozen screens. The root layout fills it once with the same
 * table it passes everywhere else; screens keep taking `strings` as a prop.
 */

import { createContext, useContext, useMemo, type ReactNode } from "react";

import { translator, type Strings, type Translate } from "@/lib/strings";

const StringsContext = createContext<Strings>({});

export function StringsProvider({ strings, children }: { strings: Strings; children: ReactNode }) {
  return <StringsContext.Provider value={strings}>{children}</StringsContext.Provider>;
}

/** The translator for the table the layout provided (keys fall back to themselves). */
export function useTranslate(): Translate {
  const strings = useContext(StringsContext);
  return useMemo(() => translator(strings), [strings]);
}
