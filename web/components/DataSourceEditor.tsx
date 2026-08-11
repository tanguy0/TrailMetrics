"use client";

/**
 * Editor for a panel's data source — *which activities* the panel is about.
 *
 * The three modes are the app's core idea: a hand-picked set, one time window, or
 * several named windows compared side by side. "Several windows" is what used to be
 * a bespoke season editor on one page and a bespoke time-scale editor on another;
 * here it is one control that every plot understands.
 */

import { useMemo, useState } from "react";

import { formatDistanceKm } from "@/lib/format";
import { CYCLING_SPORT_TYPES, RUNNING_SPORT_TYPES } from "@/lib/sport";
import type {
  ActivityFilter,
  ActivitySummary,
  DataSourceSpec,
  SourceMode,
  TimeWindow,
} from "@/lib/types";

// Running vs cycling can never be plotted together (GAP, modelled power and
// records aren't comparable between a foot split and a bike split — see
// src/usecases/resolve_panel_data.py's `_filter_family`), so this is a single
// choice, not a flat list of every sport type: pick the family first, then
// which of its sub-sports count, same rule for every plot/panel in the app.
type SportFamily = "running" | "cycling";

const FAMILY_SPORTS: Record<SportFamily, string[]> = {
  running: RUNNING_SPORT_TYPES,
  cycling: CYCLING_SPORT_TYPES,
};

// Sentinels for "this family, but every sub-sport unticked." An empty
// `sport_types` list already means "everything" server-side (every page saved
// before this control existed relies on that), so "nothing" has to be encoded
// as a non-empty, never-matching value instead — one per family, so unticking
// every box doesn't lose track of which family was showing.
const NONE_RUNNING = "__no_sport_running__";
const NONE_CYCLING = "__no_sport_cycling__";

function familyOf(sportTypes: string[]): SportFamily {
  if (sportTypes.includes(NONE_CYCLING)) return "cycling";
  if (sportTypes.includes(NONE_RUNNING)) return "running";
  // A non-empty list that is entirely cycling reads as cycling; anything else
  // (empty, or a running/mixed list) reads as running — matching the backend's
  // own default and tie-break.
  return sportTypes.length > 0 && sportTypes.every((s) => CYCLING_SPORT_TYPES.includes(s))
    ? "cycling"
    : "running";
}

function checkedSports(sportTypes: string[], family: SportFamily): Set<string> {
  if (sportTypes.includes(NONE_RUNNING) || sportTypes.includes(NONE_CYCLING)) {
    return new Set();
  }
  const familySports = FAMILY_SPORTS[family];
  if (sportTypes.length === 0) return new Set(familySports);
  return new Set(sportTypes.filter((s) => familySports.includes(s)));
}

const MODES: { value: SourceMode; label: string; hint: string }[] = [
  {
    value: "activities",
    label: "Chosen activities",
    hint: "Pick specific runs — for comparing races.",
  },
  {
    value: "window",
    label: "One time window",
    hint: "Every activity between two dates.",
  },
  {
    value: "windows",
    label: "Several time windows",
    hint: "Named periods compared side by side — seasons, training blocks.",
  },
];

interface Props {
  source: DataSourceSpec;
  onChange: (source: DataSourceSpec) => void;
  activities: ActivitySummary[];
  oldest: string | null;
  newest: string | null;
  /** Stable per editor instance, so each panel's radios form their own group. */
  groupName: string;
}

export function DataSourceEditor({
  source,
  onChange,
  activities,
  oldest,
  newest,
  groupName,
}: Props) {
  const patch = (changes: Partial<DataSourceSpec>) => onChange({ ...source, ...changes });

  return (
    <div className="source-editor">
      <SportPicker
        filters={source.filters}
        onChange={(changes) => patch({ filters: { ...source.filters, ...changes } })}
        groupName={groupName}
      />

      <div className="source-editor__modes">
        {MODES.map((mode) => (
          <label
            key={mode.value}
            className={`mode ${source.mode === mode.value ? "mode--active" : ""}`}
          >
            <input
              type="radio"
              name={`source-mode-${groupName}`}
              checked={source.mode === mode.value}
              onChange={() => patch({ mode: mode.value, ...defaultsFor(mode.value, source, oldest, newest) })}
            />
            <span className="mode__label">{mode.label}</span>
            <span className="mode__hint">{mode.hint}</span>
          </label>
        ))}
      </div>

      {source.mode === "activities" && (
        <ActivityPicker
          activities={activities}
          selected={source.activity_ids}
          onChange={(ids) => patch({ activity_ids: ids })}
        />
      )}

      {source.mode === "window" && (
        <WindowRow
          window={source.windows[0] ?? blankWindow("All history", oldest, newest)}
          onChange={(w) => patch({ windows: [w] })}
          showName={false}
        />
      )}

      {source.mode === "windows" && (
        <WindowList
          windows={source.windows}
          onChange={(windows) => patch({ windows })}
          oldest={oldest}
          newest={newest}
        />
      )}

      <Filters source={source} onChange={patch} />
    </div>
  );
}

/** Sensible starting values when switching mode, so nothing renders empty. */
function defaultsFor(
  mode: SourceMode,
  source: DataSourceSpec,
  oldest: string | null,
  newest: string | null,
): Partial<DataSourceSpec> {
  if (mode === "window" && source.windows.length !== 1) {
    return { windows: [blankWindow("All history", oldest, newest)] };
  }
  if (mode === "windows" && source.windows.length < 2) {
    return { windows: calendarYears(oldest, newest) };
  }
  return {};
}

function blankWindow(name: string, oldest: string | null, newest: string | null): TimeWindow {
  const today = new Date().toISOString().slice(0, 10);
  return {
    name,
    start: (oldest ?? `${new Date().getFullYear()}-01-01`).slice(0, 10),
    end: (newest ?? today).slice(0, 10),
  };
}

/** One window per calendar year covered by the athlete's history. */
function calendarYears(oldest: string | null, newest: string | null): TimeWindow[] {
  const first = oldest ? new Date(oldest).getUTCFullYear() : new Date().getUTCFullYear();
  const last = newest ? new Date(newest).getUTCFullYear() : new Date().getUTCFullYear();
  const windows: TimeWindow[] = [];
  for (let year = first; year <= last; year += 1) {
    windows.push({ name: String(year), start: `${year}-01-01`, end: `${year}-12-31` });
  }
  return windows;
}

function WindowRow({
  window,
  onChange,
  showName,
  onRemove,
}: {
  window: TimeWindow;
  onChange: (w: TimeWindow) => void;
  showName: boolean;
  onRemove?: () => void;
}) {
  const invalid = window.start > window.end;
  return (
    <div className="window-row">
      {showName && (
        <input
          type="text"
          value={window.name}
          placeholder="Name (e.g. Marathon block)"
          onChange={(event) => onChange({ ...window, name: event.target.value })}
        />
      )}
      <input
        type="date"
        value={window.start}
        onChange={(event) => onChange({ ...window, start: event.target.value })}
      />
      <input
        type="date"
        value={window.end}
        onChange={(event) => onChange({ ...window, end: event.target.value })}
      />
      {onRemove && (
        <button type="button" className="button button--ghost button--small" onClick={onRemove}>
          Remove
        </button>
      )}
      {invalid && <span className="note note--error">Start is after end.</span>}
    </div>
  );
}

function WindowList({
  windows,
  onChange,
  oldest,
  newest,
}: {
  windows: TimeWindow[];
  onChange: (windows: TimeWindow[]) => void;
  oldest: string | null;
  newest: string | null;
}) {
  // Overlap is legal — an activity lands in both groups — but almost always a typo.
  const overlaps = useMemo(() => {
    const pairs: string[] = [];
    for (let i = 0; i < windows.length; i += 1) {
      for (let j = i + 1; j < windows.length; j += 1) {
        const a = windows[i];
        const b = windows[j];
        if (a.start <= b.end && b.start <= a.end) pairs.push(`${a.name} ⇄ ${b.name}`);
      }
    }
    return pairs;
  }, [windows]);

  return (
    <div className="window-list">
      {windows.map((window, index) => (
        <WindowRow
          key={index}
          window={window}
          showName
          onChange={(next) => onChange(windows.map((w, i) => (i === index ? next : w)))}
          onRemove={
            windows.length > 1 ? () => onChange(windows.filter((_, i) => i !== index)) : undefined
          }
        />
      ))}
      <div className="row-actions">
        <button
          type="button"
          className="button button--ghost button--small"
          onClick={() => onChange([...windows, blankWindow(`Window ${windows.length + 1}`, oldest, newest)])}
        >
          Add window
        </button>
        <button
          type="button"
          className="button button--ghost button--small"
          onClick={() => onChange(calendarYears(oldest, newest))}
        >
          Use calendar years
        </button>
      </div>
      {overlaps.length > 0 && (
        <p className="note">Windows overlap: {overlaps.join(", ")}. Activities count in both.</p>
      )}
    </div>
  );
}

function ActivityPicker({
  activities,
  selected,
  onChange,
}: {
  activities: ActivitySummary[];
  selected: number[];
  onChange: (ids: number[]) => void;
}) {
  const [query, setQuery] = useState("");

  const shown = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const matches = needle
      ? activities.filter((a) => a.label.toLowerCase().includes(needle))
      : activities;
    // Long histories make an unbounded list unusable; selected ones always show.
    return matches.slice(0, 200);
  }, [activities, query]);

  const toggle = (id: number) =>
    onChange(selected.includes(id) ? selected.filter((v) => v !== id) : [...selected, id]);

  const selectedSet = new Set(selected);
  const chosen = activities.filter((a) => selectedSet.has(a.activity_id));

  return (
    <div className="activity-picker">
      <input
        type="search"
        placeholder="Filter by date, sport or distance…"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
      />
      <p className="muted">
        {selected.length} selected
        {chosen.length > 0 && ` · ${formatDistanceKm(chosen.reduce((sum, a) => sum + a.distance_m, 0))} total`}
      </p>
      <div className="activity-list">
        {shown.map((activity) => (
          <label key={activity.activity_id} className="activity-item">
            <input
              type="checkbox"
              checked={selectedSet.has(activity.activity_id)}
              onChange={() => toggle(activity.activity_id)}
            />
            <span>{activity.label}</span>
            {!activity.has_streams && <span className="tag">summary only</span>}
          </label>
        ))}
        {!shown.length && <p className="muted">No activity matches that filter.</p>}
      </div>
      {activities.length > shown.length && (
        <p className="muted">
          Showing {shown.length} of {activities.length} — narrow the filter to see the rest.
        </p>
      )}
    </div>
  );
}

/**
 * The one, shared sport picker: family first (running vs cycling, mutually
 * exclusive — they can never share a plot, see this file's header comment),
 * then which of that family's sub-sports count. Shown up front, unconditionally
 * — not tucked inside the collapsed `Filters` below — since which sport a panel
 * covers is a primary decision, not an advanced one.
 */
function SportPicker({
  filters,
  onChange,
  groupName,
}: {
  filters: ActivityFilter;
  onChange: (changes: Partial<ActivityFilter>) => void;
  groupName: string;
}) {
  const family = familyOf(filters.sport_types);
  const checked = checkedSports(filters.sport_types, family);

  const selectFamily = (next: SportFamily) => {
    if (next === family) return;
    // Always the explicit, full list — never empty — so the choice round-trips
    // unambiguously instead of leaning on "empty means everything."
    onChange({ sport_types: [...FAMILY_SPORTS[next]] });
  };

  const toggleSport = (sport: string) => {
    const next = new Set(checked);
    if (next.has(sport)) next.delete(sport);
    else next.add(sport);
    onChange({
      sport_types: next.size > 0
        ? [...next]
        : [family === "running" ? NONE_RUNNING : NONE_CYCLING],
    });
  };

  return (
    <div className="sport-picker">
      <div className="sport-picker__family">
        {(["running", "cycling"] as const).map((value) => (
          <label
            key={value}
            className={`mode ${family === value ? "mode--active" : ""}`}
          >
            <input
              type="radio"
              name={`sport-family-${groupName}`}
              checked={family === value}
              onChange={() => selectFamily(value)}
            />
            <span className="mode__label">{value === "running" ? "Running" : "Cycling"}</span>
          </label>
        ))}
      </div>
      <div className="multichoice">
        {FAMILY_SPORTS[family].map((sport) => (
          <label key={sport} className="multichoice__item">
            <input
              type="checkbox"
              checked={checked.has(sport)}
              onChange={() => toggleSport(sport)}
            />
            <span>{sport}</span>
          </label>
        ))}
      </div>
      {checked.size === 0 && (
        <p className="note note--error">No sport selected — nothing will be shown.</p>
      )}
    </div>
  );
}

function Filters({
  source,
  onChange,
}: {
  source: DataSourceSpec;
  onChange: (changes: Partial<DataSourceSpec>) => void;
}) {
  const filters = source.filters;
  const setFilters = (changes: Partial<DataSourceSpec["filters"]>) =>
    onChange({ filters: { ...filters, ...changes } });

  return (
    <details className="filters">
      <summary>Distance filters</summary>
      <div className="filters__body">
        <div className="param">
          <label className="param__label">Min distance (km)</label>
          <input
            type="number"
            min={0}
            value={filters.min_distance_km ?? ""}
            onChange={(event) =>
              setFilters({
                min_distance_km: event.target.value === "" ? null : Number(event.target.value),
              })
            }
          />
        </div>
        <div className="param">
          <label className="param__label">Max distance (km)</label>
          <input
            type="number"
            min={0}
            value={filters.max_distance_km ?? ""}
            onChange={(event) =>
              setFilters({
                max_distance_km: event.target.value === "" ? null : Number(event.target.value),
              })
            }
          />
        </div>
      </div>
    </details>
  );
}
