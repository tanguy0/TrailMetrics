"use client";

/**
 * Training: the diary calendar shared between an athlete and (eventually) their
 * coach — a week per row, a day per column, today's week on top when the screen
 * opens.
 *
 * Three kinds of cell: a planned workout and a planned goal are both plain text
 * (title on the calendar, body when opened) stored in `planned_items`, told apart
 * only by `kind`; a completed session is a Strava activity shown with its metadata,
 * clickable into the same `SessionDetail` Home's latest-activity widget uses.
 *
 * The calendar loads more weeks in both directions as the athlete scrolls, with no
 * hard bound either way — an `IntersectionObserver` sentinel above the earliest
 * week and one below the latest trigger the next page. Prepending older weeks
 * would otherwise yank the scroll position; the fix is the standard one: measure
 * the scroll container's height before the prepend and add the delta to `scrollTop`
 * once the new weeks are in the DOM.
 *
 * The coach/athlete permission layer is not built yet — every item is scoped to
 * the signed-in athlete like the rest of the app.
 */

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { useRouter } from "next/navigation";

import { Modal } from "@/components/Modal";
import { SessionDetail } from "@/components/SessionDetail";
import {
  ApiError,
  createPlannedItem,
  deletePlannedItem,
  getTrainingCalendar,
  renderPanel,
  setActivityRpeFeeling,
  updatePlannedItem,
} from "@/lib/api";
import {
  formatDistanceAdaptive, formatHms, formatHoursMinutes, formatNumber, formatPace, formatSpeed,
} from "@/lib/format";
import {
  CYCLING_SPORT_TYPES,
  HIKING_SPORT_TYPES,
  RUNNING_SPORT_TYPES,
  SWIMMING_SPORT_TYPES,
  sportTone,
} from "@/lib/sport";
import { translator, type Strings, type Translate } from "@/lib/strings";
import type {
  ActivityCard,
  PanelSpec,
  PlannedItem,
  PlannedItemImportance,
  PlannedItemKind,
} from "@/lib/types";

type T = Translate;
type FitnessTrend = "increasing" | "stable" | "decreasing";

const WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const MONTH_LABELS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

const INITIAL_PAST_WEEKS = 4;
const INITIAL_FUTURE_WEEKS = 1;
const PAGE_WEEKS = 4;

// Fixed display order within a day cell: note, then workout, then goal.
// Completed sessions always render last (see `DayCell`'s JSX order).
const KIND_ORDER: Record<PlannedItemKind, number> = { note: 0, workout: 1, goal: 2 };

// --- Date helpers: everything here is local-calendar-date arithmetic on plain
// "YYYY-MM-DD" strings, so there is no timezone shifting to reason about. ------

function isoDate(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

function parseIsoDate(value: string): Date {
  const [y, m, d] = value.split("-").map(Number);
  return new Date(y, m - 1, d);
}

function addDays(value: string, days: number): string {
  const date = parseIsoDate(value);
  date.setDate(date.getDate() + days);
  return isoDate(date);
}

/** The Monday on or before `value`. */
function mondayOf(value: string): string {
  const date = parseIsoDate(value);
  const offset = (date.getDay() + 6) % 7; // getDay(): 0 = Sunday .. 6 = Saturday
  return addDays(value, -offset);
}

function weekDays(weekStart: string): string[] {
  return Array.from({ length: 7 }, (_, i) => addDays(weekStart, i));
}

function rangeWeeks(fromWeekStart: string, count: number): string[] {
  return Array.from({ length: count }, (_, i) => addDays(fromWeekStart, i * 7));
}

/** Every date from `start` to `end`, inclusive. Plain string comparison sorts
 * the same as chronological order for `YYYY-MM-DD`, so this needs no Date math
 * beyond stepping one day at a time. */
function datesBetween(start: string, end: string): string[] {
  const dates: string[] = [];
  for (let date = start; date <= end; date = addDays(date, 1)) dates.push(date);
  return dates;
}

type ModalState =
  | { type: "session"; activity: ActivityCard }
  | { type: "edit"; item: PlannedItem }
  | { type: "new"; date: string }
  | { type: "rating"; activity: ActivityCard; kind: "rpe" | "feeling" };

export function TrainingScreen({ strings }: { strings: Strings }) {
  const t = translator(strings);
  const router = useRouter();

  const todayIso = isoDate(new Date());
  const todayWeekStart = mondayOf(todayIso);

  const [weekStarts, setWeekStarts] = useState<string[]>(() =>
    rangeWeeks(
      addDays(todayWeekStart, -7 * INITIAL_PAST_WEEKS),
      INITIAL_PAST_WEEKS + 1 + INITIAL_FUTURE_WEEKS,
    ),
  );
  const [plannedItems, setPlannedItems] = useState<Record<string, PlannedItem>>({});
  const [activities, setActivities] = useState<Record<number, ActivityCard>>({});
  const [error, setError] = useState<string | null>(null);
  const [modal, setModal] = useState<ModalState | null>(null);
  // Fitness on each day that's actually happened, from the earliest loaded
  // week through today — see the effect below. Keyed by ISO date, same
  // format as every other date string in this file.
  const [fitnessByDate, setFitnessByDate] = useState<Record<string, number>>({});

  const containerRef = useRef<HTMLDivElement>(null);
  const weekRefs = useRef(new Map<string, HTMLDivElement>());
  const loadingRef = useRef(false);
  const prependAdjustRef = useRef<number | null>(null);
  const scrolledToTodayRef = useRef(false);

  const fetchRange = useCallback(async (start: string, end: string): Promise<boolean> => {
    try {
      const result = await getTrainingCalendar(start, end);
      setPlannedItems((current) => {
        const next = { ...current };
        for (const item of result.planned_items) next[item.id] = item;
        return next;
      });
      setActivities((current) => {
        const next = { ...current };
        for (const activity of result.activities) next[activity.activity_id] = activity;
        return next;
      });
      return true;
    } catch (caught) {
      if (caught instanceof ApiError && caught.isUnauthorized) {
        router.push("/");
        return false;
      }
      setError((caught as Error).message);
      return false;
    }
  }, [router]);

  // Initial load: runs once. `weekStarts`'s initial value already covers the
  // range fetched here; every later change to it goes through loadPast/loadFuture,
  // which fetch their own slice.
  useEffect(() => {
    const start = weekStarts[0];
    const end = addDays(weekStarts[weekStarts.length - 1], 6);
    fetchRange(start, end);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const earliestWeekStart = weekStarts[0];

  /**
   * The Fitness trace from the same Banister model Home uses, covering every
   * day from the earliest loaded week through today — every past and
   * current week's badge (see `WeekSummary` below) is a before/after delta
   * read off this one map, not Home's 4/12-week trend line and not a
   * separate fetch per week. A future week has no data yet and gets no
   * badge; re-runs whenever scrolling back loads earlier weeks, since that
   * moves the range this needs to cover.
   *
   * `fitness_fatigue` ignores the source's sport filter and warms the model
   * up from the athlete's whole history regardless of the window (see that
   * plot's own docstring) — the window here only controls what's sliced
   * back. A day with no session already contributes zero load to the model,
   * so a quiet day before any run this week is included at its true value,
   * not skipped.
   */
  useEffect(() => {
    if (earliestWeekStart > todayIso) return;
    let live = true;
    const panel: PanelSpec = {
      id: "panel_training_fitness_trend",
      title: "",
      description: "",
      columns: 1,
      source: {
        mode: "window",
        activity_ids: [],
        selection_label: "",
        windows: [{ name: "", start: earliestWeekStart, end: todayIso }],
        filters: { sport_types: [], min_distance_km: null, max_distance_km: null },
      },
      plots: [
        { id: "plot_training_fitness_trend", plot_type: "fitness_fatigue", title: null, params: {} },
      ],
    };
    renderPanel(panel)
      .then((result) => {
        if (!live) return;
        const trace = result.panel.plots[0]?.output?.charts[0]?.traces[0];
        const byDate: Record<string, number> = {};
        (trace?.x ?? []).forEach((date, index) => {
          const value = trace?.y[index];
          if (typeof date === "string" && value != null) byDate[date] = value;
        });
        setFitnessByDate(byDate);
      })
      // Left as whatever it was rather than surfaced as an error: the
      // calendar's own data is the part that matters, and a missing badge
      // is not actionable here.
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [earliestWeekStart, todayIso]);

  // Scroll so today's week starts at the top, once it is on the page. Guarded so
  // a later week being added (top or bottom) never re-triggers it.
  useEffect(() => {
    if (scrolledToTodayRef.current) return;
    const element = weekRefs.current.get(todayWeekStart);
    if (!element) return;
    element.scrollIntoView({ block: "start" });
    scrolledToTodayRef.current = true;
  }, [weekStarts, todayWeekStart]);

  // Preserve scroll position when older weeks are prepended: the container grew
  // taller above the fold, so push scrollTop down by exactly that much.
  useLayoutEffect(() => {
    if (prependAdjustRef.current == null) return;
    const container = containerRef.current;
    if (container) {
      container.scrollTop += container.scrollHeight - prependAdjustRef.current;
    }
    prependAdjustRef.current = null;
  }, [weekStarts]);

  const loadPast = useCallback(async () => {
    if (loadingRef.current) return;
    loadingRef.current = true;
    const firstWeek = weekStarts[0];
    const newWeeks = rangeWeeks(addDays(firstWeek, -7 * PAGE_WEEKS), PAGE_WEEKS);
    const ok = await fetchRange(newWeeks[0], addDays(firstWeek, -1));
    if (ok) {
      const container = containerRef.current;
      prependAdjustRef.current = container ? container.scrollHeight : null;
      setWeekStarts((current) => [...newWeeks, ...current]);
    }
    loadingRef.current = false;
  }, [weekStarts, fetchRange]);

  const loadFuture = useCallback(async () => {
    if (loadingRef.current) return;
    loadingRef.current = true;
    const lastWeek = weekStarts[weekStarts.length - 1];
    const newWeeks = rangeWeeks(addDays(lastWeek, 7), PAGE_WEEKS);
    const ok = await fetchRange(newWeeks[0], addDays(newWeeks[newWeeks.length - 1], 6));
    if (ok) setWeekStarts((current) => [...current, ...newWeeks]);
    loadingRef.current = false;
  }, [weekStarts, fetchRange]);

  const topSentinelRef = useRef<HTMLDivElement>(null);
  const bottomSentinelRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const container = containerRef.current;
    const top = topSentinelRef.current;
    const bottom = bottomSentinelRef.current;
    if (!container || !top || !bottom) return;
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          if (entry.target === top) loadPast();
          if (entry.target === bottom) loadFuture();
        }
      },
      { root: container, threshold: 0 },
    );
    observer.observe(top);
    observer.observe(bottom);
    return () => observer.disconnect();
  }, [loadPast, loadFuture]);

  const itemsByDate: Record<string, PlannedItem[]> = {};
  for (const item of Object.values(plannedItems)) {
    // A workout or goal's `end_date` is always its own `date`, so this only
    // ever iterates more than once for a multi-day note.
    for (const date of datesBetween(item.date, item.end_date)) {
      (itemsByDate[date] ??= []).push(item);
    }
  }
  const activitiesByDate: Record<string, ActivityCard[]> = {};
  for (const activity of Object.values(activities)) {
    const date = activity.date?.slice(0, 10);
    if (date) (activitiesByDate[date] ??= []).push(activity);
  }

  const saveNewItem = async (
    date: string, kind: PlannedItemKind, title: string, body: string,
    importance: PlannedItemImportance, endDate: string,
  ) => {
    try {
      const created = await createPlannedItem({
        kind, date, end_date: endDate, title, body, importance,
      });
      setPlannedItems((current) => ({ ...current, [created.id]: created }));
      setModal(null);
    } catch (caught) {
      setError((caught as Error).message);
      throw caught;
    }
  };

  const saveEditedItem = async (
    id: string, title: string, body: string, importance: PlannedItemImportance,
    endDate: string,
  ) => {
    try {
      const updated = await updatePlannedItem(id, {
        title, body, importance, end_date: endDate,
      });
      setPlannedItems((current) => ({ ...current, [id]: updated }));
      setModal(null);
    } catch (caught) {
      setError((caught as Error).message);
      throw caught;
    }
  };

  const duplicateItem = async (item: PlannedItem) => {
    try {
      const created = await createPlannedItem({
        kind: item.kind,
        date: item.date,
        end_date: item.date,
        title: item.title,
        body: item.body,
        importance: item.importance,
      });
      setPlannedItems((current) => ({ ...current, [created.id]: created }));
      setModal(null);
    } catch (caught) {
      setError((caught as Error).message);
      throw caught;
    }
  };

  const removeItem = async (id: string) => {
    try {
      await deletePlannedItem(id);
      setPlannedItems((current) => {
        const next = { ...current };
        delete next[id];
        return next;
      });
      setModal(null);
    } catch (caught) {
      setError((caught as Error).message);
      throw caught;
    }
  };

  const saveRating = async (
    activityId: number, changes: Partial<{ rpe: number; feeling: "faible" | "ok" | "fort" }>,
  ) => {
    try {
      const updated = await setActivityRpeFeeling(activityId, changes);
      setActivities((current) => {
        const previous = current[activityId];
        if (!previous) return current;
        return {
          ...current,
          [activityId]: {
            ...previous,
            rpe: updated.rpe ?? previous.rpe,
            feeling: (updated.feeling as "faible" | "ok" | "fort" | null) ?? previous.feeling,
          },
        };
      });
      setModal(null);
    } catch (caught) {
      setError((caught as Error).message);
    }
  };

  const moveItem = async (id: string, date: string) => {
    const previous = plannedItems[id];
    if (!previous || previous.date === date) return;
    // Preserve the span length (zero for a workout or goal, which are always
    // one day) rather than just moving `date` — otherwise `end_date` stays
    // behind at its old value, which for a single-day item means it now
    // precedes `date`.
    const spanDays = Math.round(
      (parseIsoDate(previous.end_date).getTime() - parseIsoDate(previous.date).getTime())
        / 86_400_000,
    );
    const endDate = addDays(date, spanDays);
    // Optimistic: dropped onto a day, it should land there immediately. A failed
    // move is rare enough that reverting on error is simpler than blocking the drop.
    setPlannedItems((current) => (
      { ...current, [id]: { ...previous, date, end_date: endDate } }
    ));
    try {
      const updated = await updatePlannedItem(id, { date, end_date: endDate });
      setPlannedItems((current) => ({ ...current, [id]: updated }));
    } catch (caught) {
      setError((caught as Error).message);
      setPlannedItems((current) => ({ ...current, [id]: previous }));
    }
  };

  return (
    <main className="container">
      <header className="hero">
        <div>
          <h1 className="hero__name">{t("nav.training")}</h1>
        </div>
      </header>

      {error && <p className="note note--error">{error}</p>}

      <div className="training-calendar" ref={containerRef}>
        <div ref={topSentinelRef} className="training-sentinel" />
        {weekStarts.map((weekStart) => (
          <WeekRow
            key={weekStart}
            weekStart={weekStart}
            todayIso={todayIso}
            itemsByDate={itemsByDate}
            activitiesByDate={activitiesByDate}
            fitnessByDate={fitnessByDate}
            onRef={(element) => {
              if (element) weekRefs.current.set(weekStart, element);
              else weekRefs.current.delete(weekStart);
            }}
            onOpenItem={(item) => setModal({ type: "edit", item })}
            onOpenSession={(activity) => setModal({ type: "session", activity })}
            onOpenRating={(activity, kind) => setModal({ type: "rating", activity, kind })}
            onAddItem={(date) => setModal({ type: "new", date })}
            onDropItem={moveItem}
            t={t}
          />
        ))}
        <div ref={bottomSentinelRef} className="training-sentinel" />
      </div>

      {modal?.type === "new" && (
        <ItemForm
          title={t("training.new_plan_title")}
          startDate={modal.date}
          onCancel={() => setModal(null)}
          onSave={(kind, title, body, importance, endDate) =>
            saveNewItem(modal.date, kind, title, body, importance, endDate)
          }
          t={t}
        />
      )}

      {modal?.type === "edit" && (
        <ItemForm
          title={t(`training.kind.${modal.item.kind}`)}
          kind={modal.item.kind}
          startDate={modal.item.date}
          initialTitle={modal.item.title}
          initialBody={modal.item.body}
          initialImportance={modal.item.importance}
          initialEndDate={modal.item.end_date}
          onCancel={() => setModal(null)}
          onSave={(kind, title, body, importance, endDate) =>
            saveEditedItem(modal.item.id, title, body, importance, endDate)
          }
          onDelete={() => removeItem(modal.item.id)}
          onDuplicate={
            modal.item.kind === "workout" ? () => duplicateItem(modal.item) : undefined
          }
          t={t}
        />
      )}

      {modal?.type === "session" && (
        <Modal
          title={`${modal.activity.sport_type} · ${modal.activity.date?.slice(0, 10) ?? ""}`}
          onClose={() => setModal(null)}
          wide
        >
          <SessionDetail activity={modal.activity} t={t} />
        </Modal>
      )}

      {modal?.type === "rating" && (
        <Modal
          title={t(modal.kind === "rpe" ? "training.session.rpe_title" : "training.session.feeling_title")}
          onClose={() => setModal(null)}
        >
          {modal.kind === "rpe" ? (
            <div className="rpe-picker">
              {Array.from({ length: 10 }, (_, i) => i + 1).map((value) => (
                <button
                  key={value}
                  type="button"
                  className="rpe-picker__value"
                  onClick={() => saveRating(modal.activity.activity_id, { rpe: value })}
                >
                  {value}
                </button>
              ))}
            </div>
          ) : (
            <div className="feeling-picker">
              {(["faible", "ok", "fort"] as const).map((value) => (
                <button
                  key={value}
                  type="button"
                  className="feeling-picker__value"
                  onClick={() => saveRating(modal.activity.activity_id, { feeling: value })}
                >
                  {t(`training.session.feeling_${value}`)}
                </button>
              ))}
            </div>
          )}
        </Modal>
      )}
    </main>
  );
}

// --- Week / day grid ---------------------------------------------------------

function WeekRow({
  weekStart,
  todayIso,
  itemsByDate,
  activitiesByDate,
  fitnessByDate,
  onRef,
  onOpenItem,
  onOpenSession,
  onOpenRating,
  onAddItem,
  onDropItem,
  t,
}: {
  weekStart: string;
  todayIso: string;
  itemsByDate: Record<string, PlannedItem[]>;
  activitiesByDate: Record<string, ActivityCard[]>;
  fitnessByDate: Record<string, number>;
  onRef: (element: HTMLDivElement | null) => void;
  onOpenItem: (item: PlannedItem) => void;
  onOpenSession: (activity: ActivityCard) => void;
  onOpenRating: (activity: ActivityCard, kind: "rpe" | "feeling") => void;
  onAddItem: (date: string) => void;
  onDropItem: (id: string, date: string) => void;
  t: T;
}) {
  const start = parseIsoDate(weekStart);
  const days = weekDays(weekStart);

  return (
    <div className="training-week" ref={onRef}>
      <div className="training-week__label">
        {MONTH_LABELS[start.getMonth()]} {start.getDate()}
      </div>
      <div className="training-week__days">
        {days.map((date) => (
          <DayCell
            key={date}
            date={date}
            isToday={date === todayIso}
            items={itemsByDate[date] ?? []}
            sessions={activitiesByDate[date] ?? []}
            onOpenItem={onOpenItem}
            onOpenSession={onOpenSession}
            onOpenRating={onOpenRating}
            onAddItem={() => onAddItem(date)}
            onDrop={(id) => onDropItem(id, date)}
            t={t}
          />
        ))}
      </div>
      <WeekSummary
        days={days}
        todayIso={todayIso}
        activitiesByDate={activitiesByDate}
        fitnessByDate={fitnessByDate}
        t={t}
      />
    </div>
  );
}

const FEELING_SCORE = { faible: 1, ok: 2, fort: 3 } as const;
const FEELING_BY_SCORE = ["faible", "ok", "fort"] as const;

/** Green at 1 up to red at `max` (RPE: 1-10; feeling: 1-3, via FEELING_SCORE) —
 * same hue ramp for both, just a different ceiling, so a 6/10 RPE and an "ok"
 * feeling (2/3) don't have to agree on what "medium" looks like in isolation,
 * only on the same green-to-red direction. */
function ratingColor(value: number, max: number): string {
  const clamped = Math.max(1, Math.min(max, value));
  const hue = 120 * (1 - (clamped - 1) / (max - 1));
  return `hsl(${hue}, 70%, 42%)`;
}

/** Feeling gets 3 fixed brand colours rather than RPE's continuous ramp — only
 * 3 values exist, so there's no "between" shade to interpolate. "Fort" is the
 * good outcome (green), "faible" the one to flag (red) — the reverse of RPE,
 * where a high number is the demanding one. */
const FEELING_COLOR: Record<"faible" | "ok" | "fort", string> = {
  fort: "var(--primary)",
  ok: "var(--sunrise)",
  faible: "var(--danger)",
};

/**
 * Totals for the week: one card, one set of icons shared by every sport
 * (running, cycling, "other" — hiking and swimming merged, in that order) —
 * not a separate icon-and-box per sport repeating the same three icons. All
 * three always render, zeroed out on a quiet week, so the card's shape never
 * shifts week to week.
 *
 * A CSS grid, laid out explicitly by `gridColumn`/`gridRow` rather than
 * relying on source order: column 1 is the icons, columns 2–4 are the three
 * sports, column 5 (wider — see `--week-summary-detail-col-width`) is the
 * fitness trend / week-average RPE / week-average feeling, since it isn't a
 * sport total. Row 1 is the "Week summary" title (spanning every column), row
 * 2 is the sport tags, rows 3–5 are distance/climb/time for a sport column
 * and fitness/RPE/feeling for column 5 — the same three rows either way, so
 * the card's total height never changes.
 */
function WeekSummary({
  days,
  todayIso,
  activitiesByDate,
  fitnessByDate,
  t,
}: {
  days: string[];
  todayIso: string;
  activitiesByDate: Record<string, ActivityCard[]>;
  fitnessByDate: Record<string, number>;
  t: T;
}) {
  const fitnessTrend = weekFitnessTrend(days, todayIso, fitnessByDate);
  const totals = {
    run: _emptyTotals(), ride: _emptyTotals(), other: _emptyTotals(),
  };
  const rpeValues: number[] = [];
  const feelingScores: number[] = [];
  for (const date of days) {
    for (const activity of activitiesByDate[date] ?? []) {
      const bucket = RUNNING_SPORT_TYPES.includes(activity.sport_type)
        ? totals.run
        : CYCLING_SPORT_TYPES.includes(activity.sport_type)
          ? totals.ride
          : HIKING_SPORT_TYPES.includes(activity.sport_type)
              || SWIMMING_SPORT_TYPES.includes(activity.sport_type)
            ? totals.other
            : null;
      if (bucket) {
        bucket.distance_m += activity.distance_m ?? 0;
        bucket.elevation_gain_m += activity.elevation_gain_m ?? 0;
        bucket.moving_s += activity.moving_s ?? 0;
        bucket.count += 1;
      }
      if (activity.rpe != null) rpeValues.push(activity.rpe);
      if (activity.feeling != null) feelingScores.push(FEELING_SCORE[activity.feeling]);
    }
  }
  const avgRpe = rpeValues.length
    ? rpeValues.reduce((a, b) => a + b, 0) / rpeValues.length
    : null;
  const avgFeelingScore = feelingScores.length
    ? Math.round(feelingScores.reduce((a, b) => a + b, 0) / feelingScores.length)
    : null;
  const avgFeeling = avgFeelingScore != null ? FEELING_BY_SCORE[avgFeelingScore - 1] : null;

  // Always all 3 — a quiet week reads as zeros in its sport's own colour, not as
  // a column disappearing, so the card's shape never shifts week to week.
  const columns = [
    { tone: "running", totals: totals.run, label: t("training.week.running") },
    { tone: "cycling", totals: totals.ride, label: t("training.week.cycling") },
    { tone: "other", totals: totals.other, label: t("training.week.other") },
  ] as const;

  return (
    <div className="training-week__summary">
      <div className="week-summary">
        <div className="week-summary__title" style={{ gridColumn: "1 / -1", gridRow: 1 }}>
          <span className="week-summary__title-text">{t("training.week.summary_title")}</span>
        </div>

        <span className="week-summary__icon" style={{ gridColumn: 1, gridRow: 3 }} aria-hidden="true">
          📏
        </span>
        <span className="week-summary__icon" style={{ gridColumn: 1, gridRow: 4 }} aria-hidden="true">
          ⛰️
        </span>
        <span className="week-summary__icon" style={{ gridColumn: 1, gridRow: 5 }} aria-hidden="true">
          ⏱️
        </span>

        {columns.map((column, index) => (
          <SportColumn
            key={column.tone}
            tone={column.tone}
            totals={column.totals}
            label={column.label}
            gridColumn={index + 2}
          />
        ))}

        <WeekDetailColumn
          gridColumn={5}
          fitnessTrend={fitnessTrend}
          avgRpe={avgRpe}
          avgFeeling={avgFeeling}
          t={t}
        />
      </div>
    </div>
  );
}

function _emptyTotals() {
  return { distance_m: 0, elevation_gain_m: 0, moving_s: 0, count: 0 };
}

/**
 * A week's fitness delta: `fitnessByDate` at the week's last day that's
 * actually happened (today, for the current week; the week's own Sunday,
 * for a past one) minus `fitnessByDate` at its first day — `null` for a
 * week that hasn't started yet, or one `fitnessByDate` doesn't cover (it
 * only reaches back to the earliest loaded week; see the effect in
 * `TrainingScreen`). ISO date strings compare correctly with `<=` — no need
 * to parse them into `Date`s just to order them.
 */
function weekFitnessTrend(
  days: string[],
  todayIso: string,
  fitnessByDate: Record<string, number>,
): FitnessTrend | null {
  const weekStart = days[0];
  if (weekStart > todayIso) return null;
  const weekEnd = days[days.length - 1];
  const effectiveEnd = weekEnd <= todayIso ? weekEnd : todayIso;
  const startValue = fitnessByDate[weekStart];
  const endValue = fitnessByDate[effectiveEnd];
  if (startValue == null || endValue == null) return null;
  const delta = endValue - startValue;
  return delta > 1 ? "increasing" : delta < -1 ? "decreasing" : "stable";
}

/** One sport's tag and three values, placed into the shared grid's `gridColumn`.
 * `display: contents` on the wrapper so the tag/value spans become the actual
 * grid items (each still needs its own `gridRow`) while still letting one
 * `week-summary__col--{tone}` class color all four through inheritance. */
function SportColumn({
  tone,
  totals,
  label,
  gridColumn,
}: {
  tone: "running" | "cycling" | "other";
  totals: { distance_m: number; elevation_gain_m: number; moving_s: number };
  label: string;
  gridColumn: number;
}) {
  return (
    <div className={`week-summary__col week-summary__col--${tone}`} style={{ display: "contents" }}>
      <span className="week-summary__tag" style={{ gridColumn, gridRow: 2 }}>
        {label}
      </span>
      <span className="week-summary__value" style={{ gridColumn, gridRow: 3 }}>
        {formatDistanceAdaptive(totals.distance_m / 1000)} km
      </span>
      <span className="week-summary__value" style={{ gridColumn, gridRow: 4 }}>
        {formatNumber(totals.elevation_gain_m, 0)} m
      </span>
      <span className="week-summary__value" style={{ gridColumn, gridRow: 5 }}>
        {formatHoursMinutes(totals.moving_s)}
      </span>
    </div>
  );
}

/** Column 5, always present regardless of how many sport columns this week has —
 * it isn't a sport total, so it doesn't compete for the same slots. Reuses the
 * same three rows (3-5) a sport column's stat block occupies, so the card's
 * total size never changes: the fitness trend (an arrow now, not a word — the
 * word is still there as a tooltip), the week's average RPE (coloured via
 * `ratingColor`), and its average feeling (see FEELING_SCORE/FEELING_BY_SCORE
 * — averaged as a number, rounded, then mapped back to a tag, coloured via
 * `FEELING_COLOR`), each shown as "—" when the week has nothing yet. */
function WeekDetailColumn({
  gridColumn,
  fitnessTrend,
  avgRpe,
  avgFeeling,
  t,
}: {
  gridColumn: number;
  fitnessTrend: FitnessTrend | null;
  avgRpe: number | null;
  avgFeeling: "faible" | "ok" | "fort" | null;
  t: T;
}) {
  const arrow = fitnessTrend === "increasing" ? "↑" : fitnessTrend === "decreasing" ? "↓" : fitnessTrend === "stable" ? "→" : null;
  return (
    <div className="week-summary__col week-summary__col--detail" style={{ display: "contents" }}>
      <span className="week-summary__value" style={{ gridColumn, gridRow: 3 }}>
        {arrow ? (
          <span
            className={`trend-badge trend-badge--${fitnessTrend} trend-badge--compact`}
            title={t(`training.week.fitness_${fitnessTrend}`)}
          >
            {t("training.week.fitness_label")} {arrow}
          </span>
        ) : "—"}
      </span>
      <span
        className="week-summary__value"
        style={{ gridColumn, gridRow: 4, ...(avgRpe != null ? { color: ratingColor(avgRpe, 10) } : {}) }}
      >
        {avgRpe != null ? avgRpe.toFixed(1) : "—"}
      </span>
      <span
        className="week-summary__value"
        style={{ gridColumn, gridRow: 5, ...(avgFeeling != null ? { color: FEELING_COLOR[avgFeeling] } : {}) }}
      >
        {avgFeeling != null ? t(`training.session.feeling_${avgFeeling}`) : "—"}
      </span>
    </div>
  );
}

function DayCell({
  date,
  isToday,
  items,
  sessions,
  onOpenItem,
  onOpenSession,
  onOpenRating,
  onAddItem,
  onDrop,
  t,
}: {
  date: string;
  isToday: boolean;
  items: PlannedItem[];
  sessions: ActivityCard[];
  onOpenItem: (item: PlannedItem) => void;
  onOpenSession: (activity: ActivityCard) => void;
  onOpenRating: (activity: ActivityCard, kind: "rpe" | "feeling") => void;
  onAddItem: () => void;
  onDrop: (id: string) => void;
  t: T;
}) {
  const [dragOver, setDragOver] = useState(false);
  const day = parseIsoDate(date);
  const weekdayIndex = (day.getDay() + 6) % 7;

  return (
    <div
      className={
        "training-day" +
        (isToday ? " training-day--today" : "") +
        (dragOver ? " training-day--drag-over" : "")
      }
      onDragOver={(event) => {
        event.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragOver(false);
        const id = event.dataTransfer.getData("text/plain");
        if (id) onDrop(id);
      }}
    >
      <div className="training-day__header">
        <span className="training-day__weekday">{WEEKDAY_LABELS[weekdayIndex]}</span>
        <span className="training-day__number">{day.getDate()}</span>
      </div>

      <div className="training-day__items">
        {[...items].sort((a, b) => KIND_ORDER[a.kind] - KIND_ORDER[b.kind]).map((item) => {
          const spansDays = item.end_date !== item.date;
          return (
            <div
              key={item.id}
              className={
                `training-pill training-pill--${item.kind}` +
                (item.kind === "goal" ? ` training-pill--${item.importance}` : "")
              }
              title={spansDays ? `${item.date} → ${item.end_date}` : undefined}
              // A multi-day note moves as a block or not at all — dragging one day
              // of it would either shift the whole span or desync `date` from
              // `end_date`, and neither has an obvious drop target. Simpler to
              // just not offer it.
              draggable={!spansDays}
              onDragStart={(event) => {
                event.dataTransfer.setData("text/plain", item.id);
                event.dataTransfer.effectAllowed = "move";
              }}
              onClick={() => onOpenItem(item)}
            >
              <span className="card-badge">
                {t(item.kind === "note" ? "training.badge.note" : "training.badge.planned")}
              </span>
              {item.title || t(`training.kind.${item.kind}`)}
            </div>
          );
        })}

        {sessions.map((activity) => {
          const km = activity.distance_m != null ? activity.distance_m / 1000 : null;
          const isCycling = CYCLING_SPORT_TYPES.includes(activity.sport_type);
          const pace = km && km > 0 && activity.moving_s != null
            ? activity.moving_s / km : null;
          const speedKmh = activity.moving_s && activity.moving_s > 0 && activity.distance_m != null
            ? (activity.distance_m / activity.moving_s) * 3.6 : null;
          return (
            <div
              key={activity.activity_id}
              role="button"
              tabIndex={0}
              className={`training-session training-session--${sportTone(activity.sport_type)}`}
              onClick={() => onOpenSession(activity)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") onOpenSession(activity);
              }}
            >
              <span className="card-badge">{t("training.badge.completed")}</span>
              <span className="training-session__sport">{activity.sport_type}</span>
              <span className="training-session__stats">
                {formatHms(activity.moving_s)} ·{" "}
                {km != null ? `${formatNumber(km, 1)} km` : "—"} ·{" "}
                {isCycling ? formatSpeed(speedKmh) : formatPace(pace)}
              </span>
              <span className="training-session__tags">
                <button
                  type="button"
                  className={`session-tag session-tag--rpe${activity.rpe != null ? " session-tag--set" : ""}`}
                  style={activity.rpe != null
                    ? { background: ratingColor(activity.rpe, 10), borderColor: ratingColor(activity.rpe, 10) }
                    : undefined}
                  onClick={(event) => {
                    event.stopPropagation();
                    onOpenRating(activity, "rpe");
                  }}
                >
                  {activity.rpe ?? t("training.session.rpe_short")}
                </button>
                <button
                  type="button"
                  className={`session-tag session-tag--feeling${activity.feeling != null ? " session-tag--set" : ""}`}
                  style={activity.feeling != null
                    ? {
                      background: FEELING_COLOR[activity.feeling],
                      borderColor: FEELING_COLOR[activity.feeling],
                    }
                    : undefined}
                  onClick={(event) => {
                    event.stopPropagation();
                    onOpenRating(activity, "feeling");
                  }}
                >
                  {activity.feeling != null
                    ? t(`training.session.feeling_${activity.feeling}`)
                    : t("training.session.feeling_short")}
                </button>
              </span>
            </div>
          );
        })}
      </div>

      <div className="training-day__add">
        <button type="button" className="training-add-btn" onClick={onAddItem}>
          {t("training.add_plan")}
        </button>
      </div>
    </div>
  );
}

// --- Item editor --------------------------------------------------------------

const PLANNED_ITEM_KINDS: PlannedItemKind[] = ["workout", "goal", "note"];

function ItemForm({
  title,
  kind: fixedKind,
  startDate,
  initialTitle = "",
  initialBody = "",
  initialImportance = "primary",
  initialEndDate,
  onCancel,
  onSave,
  onDelete,
  onDuplicate,
  t,
}: {
  title: string;
  /** Fixed once an item exists — only a brand-new one lets you pick a kind. */
  kind?: PlannedItemKind;
  startDate: string;
  initialTitle?: string;
  initialBody?: string;
  initialImportance?: PlannedItemImportance;
  initialEndDate?: string;
  onCancel: () => void;
  onSave: (
    kind: PlannedItemKind,
    title: string,
    body: string,
    importance: PlannedItemImportance,
    endDate: string,
  ) => Promise<void>;
  onDelete?: () => Promise<void>;
  onDuplicate?: () => Promise<void>;
  t: T;
}) {
  const [draftKind, setDraftKind] = useState<PlannedItemKind>(fixedKind ?? "workout");
  const [draftTitle, setDraftTitle] = useState(initialTitle);
  const [draftBody, setDraftBody] = useState(initialBody);
  const [draftImportance, setDraftImportance] =
    useState<PlannedItemImportance>(initialImportance);
  const [draftEndDate, setDraftEndDate] = useState(initialEndDate ?? startDate);
  const [busy, setBusy] = useState(false);

  const isNote = draftKind === "note";

  return (
    <Modal title={title} onClose={onCancel}>
      <div className="training-form">
        {!fixedKind && (
          <div className="training-form__kind">
            {PLANNED_ITEM_KINDS.map((option) => (
              <button
                key={option}
                type="button"
                className={
                  "training-form__kind-btn" +
                  (draftKind === option ? " training-form__kind-btn--active" : "")
                }
                onClick={() => setDraftKind(option)}
              >
                {t(`training.kind.${option}`)}
              </button>
            ))}
          </div>
        )}
        <input
          className="training-form__title"
          value={draftTitle}
          onChange={(event) => setDraftTitle(event.target.value)}
          placeholder={t("training.form.title_placeholder")}
          autoFocus
        />
        <textarea
          className="training-form__body"
          value={draftBody}
          onChange={(event) => setDraftBody(event.target.value)}
          placeholder={t("training.form.body_placeholder")}
          rows={5}
        />
        {draftKind === "goal" && (
          <div className="training-form__importance">
            <label>
              <input
                type="radio"
                name="importance"
                checked={draftImportance === "primary"}
                onChange={() => setDraftImportance("primary")}
              />
              {t("training.form.importance_primary")}
            </label>
            <label>
              <input
                type="radio"
                name="importance"
                checked={draftImportance === "secondary"}
                onChange={() => setDraftImportance("secondary")}
              />
              {t("training.form.importance_secondary")}
            </label>
          </div>
        )}
        {isNote && (
          <label className="training-form__end-date">
            {t("training.form.end_date_label")}
            <input
              type="date"
              value={draftEndDate}
              min={startDate}
              onChange={(event) => setDraftEndDate(event.target.value)}
            />
          </label>
        )}
        <div className="training-form__actions">
          {onDelete && (
            <button
              type="button"
              className="button button--ghost"
              disabled={busy}
              onClick={async () => {
                setBusy(true);
                try {
                  await onDelete();
                } catch {
                  setBusy(false);
                }
              }}
            >
              {t("training.form.delete")}
            </button>
          )}
          {onDuplicate && (
            <button
              type="button"
              className="button button--ghost"
              disabled={busy}
              onClick={async () => {
                setBusy(true);
                try {
                  await onDuplicate();
                } catch {
                  setBusy(false);
                }
              }}
            >
              {t("training.form.duplicate")}
            </button>
          )}
          <button
            type="button"
            className="button"
            disabled={busy || !draftTitle.trim()}
            onClick={async () => {
              setBusy(true);
              try {
                await onSave(
                  draftKind,
                  draftTitle.trim(),
                  draftBody,
                  draftImportance,
                  // Locking end date to start date whenever the kind isn't (or
                  // no longer is, after switching away from) a note is what
                  // keeps a workout or goal single-day even if a stale draft
                  // end date is still sitting in state.
                  isNote ? draftEndDate : startDate,
                );
              } catch {
                setBusy(false);
              }
            }}
          >
            {t("training.form.save")}
          </button>
        </div>
      </div>
    </Modal>
  );
}
