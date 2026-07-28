/**
 * Browser-side API client.
 *
 * Everything goes through `/api/proxy/...` on this origin, so there is no CORS, no
 * API URL in the bundle, and the session travels as a first-party cookie the
 * proxy turns into a bearer token.
 */

"use client";

import type {
  ActivitySummary,
  Athlete,
  HomeSummary,
  PageSpec,
  PageSummary,
  PanelResult,
  PanelSpec,
  Registry,
  RouteResult,
  SyncStatus,
  UiStrings,
} from "./types";

const LANG = process.env.NEXT_PUBLIC_LANG || "en";

export class ApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
  }

  /** The one error worth branching on: the session is gone, so re-authenticate. */
  get isUnauthorized(): boolean {
    return this.status === 401;
  }
}

async function request<T>(
  path: string,
  init: RequestInit & { query?: Record<string, string> } = {},
): Promise<T> {
  const url = new URL(`/api/proxy${path}`, window.location.origin);
  url.searchParams.set("lang", LANG);
  for (const [key, value] of Object.entries(init.query ?? {})) {
    url.searchParams.set(key, value);
  }

  const response = await fetch(url, {
    ...init,
    headers: {
      ...(init.body ? { "content-type": "application/json" } : {}),
      ...init.headers,
    },
  });

  if (!response.ok) {
    // FastAPI puts the useful message in `detail`; fall back to raw text.
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) message = typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail);
    } catch {
      /* keep the status line */
    }
    throw new ApiError(response.status, message);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

// --- Registry --------------------------------------------------------------

let registryCache: Promise<Registry> | null = null;

/** The plot catalogue. Static per language, so fetched once per page load. */
export function getRegistry(): Promise<Registry> {
  registryCache ??= request<Registry>("/registry");
  return registryCache;
}

// --- UI strings ------------------------------------------------------------

let uiStringsCache: Promise<UiStrings> | null = null;

/** The app's wording, translated. Static per language, so fetched once. */
export function getUiStrings(): Promise<UiStrings> {
  uiStringsCache ??= request<UiStrings>("/ui-strings");
  return uiStringsCache;
}

// --- Athlete ---------------------------------------------------------------

export const getAthlete = () => request<Athlete>("/auth/me");

/**
 * Patch the athlete's own body fields.
 *
 * Only the keys passed are sent, and the server leaves absent keys alone — so the
 * three widgets can be edited independently without overwriting each other. An
 * explicit `null` still clears a field.
 */
export const updateProfile = (
  changes: Partial<Pick<Athlete, "weight_kg" | "birthdate" | "height_cm">>,
) =>
  request<Athlete>("/auth/me", {
    method: "PATCH",
    body: JSON.stringify(changes),
  });

// --- Home ------------------------------------------------------------------

export const getHomeSummary = () => request<HomeSummary>("/home/summary");

/**
 * The latest activity's route. Separate from the summary because it may have to
 * call Strava for an activity imported before routes were stored, and the cards
 * must not wait on that.
 */
export const getLastActivityRoute = () =>
  request<RouteResult>("/home/last-activity/route");

// --- Activities ------------------------------------------------------------

export const listActivities = (limit = 0) =>
  request<{ activities: ActivitySummary[]; total: number }>("/activities", {
    query: limit ? { limit: String(limit) } : {},
  });

export const startSync = (options: { force?: boolean; max_activities?: number } = {}) =>
  request<{ status: string }>("/activities/sync", {
    method: "POST",
    body: JSON.stringify(options),
  });

export const getSyncStatus = () => request<SyncStatus>("/activities/sync");

// --- Pages -----------------------------------------------------------------

export const listPages = () => request<{ pages: PageSummary[] }>("/pages");

export const listBuiltinPages = () =>
  request<{ pages: PageSummary[] }>("/pages/builtin");

export const getPage = (id: string) => request<PageSpec>(`/pages/${id}`);

export const getBuiltinPage = (key: string) =>
  request<PageSpec>(`/pages/builtin/${key}`);

export const createPage = (name: string) =>
  request<PageSpec>("/pages", { method: "POST", body: JSON.stringify({ name }) });

export const savePage = (id: string, spec: PageSpec) =>
  request<PageSpec>(`/pages/${id}`, {
    method: "PUT",
    body: JSON.stringify({ spec }),
  });

export const deletePage = (id: string) =>
  request<void>(`/pages/${id}`, { method: "DELETE" });

export const duplicatePage = (id: string, name?: string) =>
  request<PageSpec>(`/pages/${id}/duplicate`, {
    method: "POST",
    body: JSON.stringify({ name: name ?? null }),
  });

export const duplicateBuiltin = (key: string, name?: string) =>
  request<PageSpec>(`/pages/builtin/${key}/duplicate`, {
    method: "POST",
    body: JSON.stringify({ name: name ?? null }),
  });

// --- Rendering -------------------------------------------------------------

export const renderPage = (spec: PageSpec, forcePlotIds: string[] = []) =>
  request<{ page_id: string; panels: PanelResult[] }>("/render", {
    method: "POST",
    body: JSON.stringify({ spec, force_plot_ids: forcePlotIds }),
  });

/** What the editor calls on every change: re-render just the panel being edited. */
export const renderPanel = (
  panel: PanelSpec,
  forcePlotIds: string[] = [],
  signal?: AbortSignal,
) =>
  request<{ panel: PanelResult }>("/render/panel", {
    method: "POST",
    body: JSON.stringify({ panel, force_plot_ids: forcePlotIds }),
    signal,
  });
