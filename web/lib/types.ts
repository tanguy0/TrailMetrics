/**
 * Wire types, mirroring the Python payloads exactly.
 *
 * Field names stay `snake_case` all the way into the browser. Consistency with the
 * backend is worth more than JavaScript convention here: it removes an entire class
 * of mapping bug, and these shapes are generated from the domain rather than
 * hand-written on both sides.
 */

// --- Specs (what a page *is*) ---------------------------------------------

export type SourceMode = "activities" | "window" | "windows";

export interface TimeWindow {
  name: string;
  start: string; // ISO date
  end: string;
}

export interface ActivityFilter {
  sport_types: string[];
  min_distance_km: number | null;
  max_distance_km: number | null;
}

export interface DataSourceSpec {
  mode: SourceMode;
  activity_ids: number[];
  selection_label: string;
  windows: TimeWindow[];
  filters: ActivityFilter;
}

export interface PlotSpec {
  id: string;
  plot_type: string;
  params: Record<string, unknown>;
  title: string | null;
}

export interface PanelSpec {
  id: string;
  title: string;
  description: string;
  source: DataSourceSpec;
  plots: PlotSpec[];
  columns: number;
}

export interface PageSpec {
  schema_version: number;
  id: string;
  name: string;
  description: string;
  icon: string;
  /**
   * Which default analysis this is, or `null` for one the athlete created.
   *
   * Set on the three analyses everyone starts with. They are stored, editable pages
   * like any other; the key means only that they cannot be deleted.
   */
  builtin_key: string | null;
  panels: PanelSpec[];
}

export interface PageSummary {
  id: string;
  name: string;
  description: string;
  icon: string;
  builtin_key: string | null;
  /** Ships with the app: editable, but not deletable. */
  is_default: boolean;
  panel_count: number;
  plot_count: number;
}

// --- Parameter schema ------------------------------------------------------

export type ParamKind =
  | "bool"
  | "int"
  | "float"
  | "text"
  /** Multi-line text: a paragraph, not a label. */
  | "textarea"
  /** An image URL, paired with an upload control. */
  | "image"
  | "choice"
  | "multichoice"
  | "group"
  | "list";

export interface Choice {
  value: string;
  label: string;
}

/** Serializable predicate; see `lib/conditions.ts` for the evaluator. */
export interface Condition {
  op: string;
  key?: string;
  value?: unknown;
  conditions?: Condition[];
}

export interface ParamSpec {
  key: string;
  kind: ParamKind;
  label: string;
  default: unknown;
  choices?: Choice[];
  choices_from?: string;
  min?: number;
  max?: number;
  step?: number;
  max_items?: number;
  help?: string;
  children?: ParamSpec[];
  visible_when?: Condition;
}

export interface PlotDefinition {
  key: string;
  label: string;
  description: string;
  category: string;
  level: "activity" | "stream" | "split";
  series_level: "group" | "activity";
  requires_streams: boolean;
  requires_weight: boolean;
  /** False for content blocks (prose, an image): they read no activity data. */
  requires_data: boolean;
  cost: "cheap" | "expensive";
  params: ParamSpec[];
}

export interface MetricInfo {
  key: string;
  label: string;
  unit: string;
  value_kind: "number" | "duration" | "pace" | "count";
  decimals: number;
  default_agg: string;
  /** Empty means the metric fixes its own aggregation, so hide the control. */
  allowed_aggs: string[];
  higher_is_better: boolean | null;
  needs_streams: boolean;
  needs_weight: boolean;
}

export interface Registry {
  plots: PlotDefinition[];
  metrics: Record<string, MetricInfo>;
  providers: Record<string, Choice[]>;
}

// --- Chart IR --------------------------------------------------------------

export type TraceKind = "line" | "step" | "bar" | "scatter" | "area";
export type AxisKind = "linear" | "date" | "duration" | "category";

export interface Axis {
  title: string;
  kind: AxisKind;
  reversed: boolean;
  tick_format: string | null;
  suffix: string | null;
  range: number[] | null;
  dtick: number | null;
  /** Tints the axis to its series; set on dual-axis charts. */
  color: string | null;
}

export interface Trace {
  name: string;
  x: (number | string | null)[];
  y: (number | null)[];
  kind: TraceKind;
  color: string | null;
  /** Which y-axis this series is measured against; only used when `y2_axis` is set. */
  axis: "y" | "y2";
  dash: string;
  width: number;
  markers: boolean;
  marker_size: number;
  opacity: number;
  stack_group: string | null;
  band_upper: (number | null)[] | null;
  band_lower: (number | null)[] | null;
  hover_text: string[] | null;
  hover_template: string | null;
  legend_group: string | null;
  show_legend: boolean;
}

/**
 * A shaded vertical slab behind the traces — one week, a race, a training block.
 *
 * Colours a stretch of x and always spans the full height, so it says nothing
 * about y. It carries no label of its own: the legend or the caption has to.
 */
export interface Band {
  x0: number | string;
  x1: number | string;
  color: string;
  opacity: number;
}

/**
 * A small tag pinned above the traces at one x position — the chart twin of
 * `.trend-badge`: coloured ink on a pale fill, read as text, not as a colour.
 * The chart leaves it room via `y_axis.range`; corners are square here, where
 * the CSS pill's are round.
 */
export interface Badge {
  x: number | string;
  text: string;
  color: string;
  fill: string | null;
  /**
   * What to draw instead when the row is too tight for `text` — thirty weekly
   * tags on a phone. Annotations don't collide-hide, so the renderer measures
   * the figure and picks; the full wording stays in the hover.
   */
  short: string | null;
}

export interface ChartData {
  title: string;
  x_axis: Axis;
  y_axis: Axis;
  /**
   * A right-hand axis, present only when the chart carries two units at once
   * (distance and climb per week, heart rate against pace). `null` keeps the
   * figure single-axis, which is the normal case.
   */
  y2_axis: Axis | null;
  traces: Trace[];
  /** Shaded x-stretches behind the traces, and the row of tags above them. */
  bands: Band[];
  badges: Badge[];
  height: number;
  hover_mode: string;
  caption: string | null;
}

export interface CellFormat {
  kind: string;
  decimals: number;
  suffix: string;
}

export interface Column {
  key: string;
  label: string;
  format: CellFormat;
  highlight: "max" | "min" | null;
}

export interface TableData {
  title: string;
  columns: Column[];
  rows: Record<string, unknown>[];
  download_name: string;
  caption: string | null;
}

/**
 * Prose inside a panel.
 *
 * The one string in the app that arrives untranslated: it is what the athlete
 * typed, not something `src/translations.py` knows about.
 */
export interface TextBlock {
  text: string;
  variant: "body" | "lede" | "heading" | "quote";
  align: "left" | "center";
  tone: "none" | "forest" | "terracotta" | "sunrise" | "plum";
}

/** An image in a panel. `src` is an external URL or `/api/proxy/assets/{id}`. */
export interface ImageBlock {
  src: string;
  alt: string;
  caption: string | null;
  /** Share of the panel's width, 10–100. */
  width_pct: number;
  align: "left" | "center";
}

export interface PlotOutput {
  charts: ChartData[];
  tables: TableData[];
  notes: string[];
  texts: TextBlock[];
  images: ImageBlock[];
}

// --- Render results --------------------------------------------------------

export interface PlotResult {
  plot_id: string;
  plot_type: string;
  title: string | null;
  params: Record<string, unknown>;
  error: string | null;
  pending: boolean;
  cost: string;
  output: PlotOutput;
}

export interface PanelResult {
  panel_id: string;
  title: string;
  description: string;
  columns: number;
  error: string | null;
  groups: { label: string; index: number; size: number }[];
  activity_count: number;
  plots: PlotResult[];
}

// --- Athlete & activities --------------------------------------------------

export interface SyncStatus {
  status: "idle" | "running" | "done" | "error";
  done: number;
  total: number;
  message: string;
  last_synced_at: string | null;
}

/**
 * Progress of the background pass that fits the expensive plots.
 *
 * Same shape as `SyncStatus`, because it is the same pattern: work too long for one
 * request, started by the client and polled.
 */
export interface PrecomputeStatus {
  status: "idle" | "running" | "done" | "error";
  done: number;
  total: number;
  message: string;
  finished_at: string | null;
}

/** One uploaded image, as `POST /assets` returns it. */
export interface AssetUpload {
  id: string;
  content_type: string;
  byte_size: number;
  /** What an image block's `src` should hold. */
  url: string;
}

export interface Athlete {
  id: number;
  firstname: string;
  lastname: string;
  display_name: string;
  profile_url: string | null;
  weight_kg: number | null;
  // Self-reported: Strava's API carries none of these. `age` is derived from
  // `birthdate` server-side so every client agrees on it.
  birthdate: string | null; // ISO date
  height_cm: number | null;
  email: string | null;
  /** Self-reported training zones and VMA pace — display-only, fed into no
   * calculation anywhere in the app. */
  hr_zone1_end: number | null;
  hr_zone2_end: number | null;
  hr_zone3_end: number | null;
  hr_zone4_end: number | null;
  hr_max: number | null;
  vma_pace_s_per_km: number | null;
  /** The athlete's chosen UI language — "en" or "fr". Always set. */
  lang: string;
  /** Server's verdict on whether the email question has been answered. */
  needs_email: boolean;
  age: number | null;
  activity_count: number;
  sport_types: string[];
  oldest_activity: string | null;
  newest_activity: string | null;
  sync: SyncStatus;
  /** Whether the *signed-in* account (not this one) is a coach account. */
  is_coach: boolean;
  /** True when a coach is browsing this account rather than their own. */
  viewing_as: boolean;
  /** Whether *this* account is the one allowed to write blog posts. */
  is_master: boolean;
}

/** One entry in a coach's athlete switcher — not the full profile. */
export interface CoachAthlete {
  id: number;
  display_name: string;
  profile_url: string | null;
}

export interface ActivitySummary {
  activity_id: number;
  start_date: string;
  sport_type: string;
  has_streams: boolean;
  distance_m: number;
  moving_s: number;
  label: string;
}

// --- Home screen -----------------------------------------------------------

/** One activity as the Home widgets show it. Raw units; the browser formats. */
export interface ActivityCard {
  activity_id: number;
  date: string | null;
  sport_type: string;
  has_streams: boolean;
  distance_m: number | null;
  elevation_gain_m: number | null;
  moving_s: number | null;
  avg_hr: number | null;
  avg_power_w: number | null;
  power_source: "measured" | "estimated" | null;
  /** Athlete-entered, not from Strava — null until set from the Training calendar. */
  rpe: number | null;
  feeling: "faible" | "ok" | "fort" | null;
}

export interface ActivityComment {
  id: string;
  activity_id: number;
  body: string;
  created_at: string;
  updated_at: string;
}

export interface HomeProfile {
  activity_count: number;
  oldest_activity: string | null;
  newest_activity: string | null;
  total_distance_m: number;
  total_elevation_gain_m: number;
  total_moving_s: number;
  furthest_activity: ActivityCard | null;
  longest_activity: ActivityCard | null;
}

export interface HomeHealth {
  age: number | null;
  birthdate: string | null;
  weight_kg: number | null;
  height_cm: number | null;
  experience_years: number | null;
  first_activity: string | null;
}

/** Fastest stored effort at one distance. Absent entirely when never covered. */
export interface HomeRecord {
  label: string;
  seconds: number;
  set_on: string | null;
  activity_id: number;
}

export interface HomeSummary {
  profile: HomeProfile;
  health: HomeHealth;
  records: HomeRecord[];
  last_activity: ActivityCard | null;
}

/**
 * The latest activity's route.
 *
 * `source` says where it came from: `stored` from the database, `strava` fetched
 * just now and cached, `none` when the activity has no route (treadmill, manual
 * entry), `unavailable` when Strava could not be reached.
 */
export interface RouteResult {
  activity_id: number | null;
  points: [number, number][];
  source: "stored" | "strava" | "none" | "unavailable";
}

// --- Training --------------------------------------------------------------

export type PlannedItemKind = "workout" | "goal" | "note";
/** Only meaningful for a goal: a secondary goal keeps the goal colour but shaded. */
export type PlannedItemImportance = "primary" | "secondary";

/** A planned workout, goal, or note on the training calendar. Title is what
 * shows on the calendar cell; body is the text revealed when the item is
 * opened. `end_date` is always present (the server defaults it to `date`) —
 * only a note is ever created with one past that, spanning every day up to and
 * including it. */
export interface PlannedItem {
  id: string;
  kind: PlannedItemKind;
  date: string; // ISO date
  end_date: string; // ISO date, >= date
  title: string;
  body: string;
  importance: PlannedItemImportance;
}

/** Everything the calendar draws for one requested date range. */
export interface TrainingCalendar {
  planned_items: PlannedItem[];
  activities: ActivityCard[];
}

// --- UI strings ------------------------------------------------------------

/**
 * The app's own wording, translated server-side and keyed without the `ui.`
 * prefix — `strings["nav.home"]`. There is no translation table in the browser;
 * adding a language in `src/translations.py` covers the whole product.
 */
export interface UiStrings {
  lang: string;
  languages: Record<string, string>;
  strings: Record<string, string>;
}

// --- Blog --------------------------------------------------------------------

/** One card in the public blog index. */
export interface BlogPostSummary {
  id: string;
  slug: string;
  title: string;
  excerpt: string;
  cover_url: string | null;
  page_count: number;
  created_at: string | null;
  /** Only present on `/blog/admin` — the public index omits it (always true there). */
  published?: boolean;
}

/** One full article: the carousel is `page_urls`, in reading order. */
export interface BlogPost {
  id: string;
  slug: string;
  title: string;
  body_text: string;
  page_urls: string[];
  page_count: number;
  published: boolean;
  created_at: string | null;
  updated_at: string | null;
}
