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
  /** Fixed ticks with their own words, for an ordinal scale. Both or neither. */
  tick_values?: number[] | null;
  tick_labels?: string[] | null;
}

export interface Trace {
  name: string;
  x: (number | string | null)[];
  y: (number | null)[];
  kind: TraceKind;
  color: string | null;
  /** Which y-axis this series is measured against; only used when `y2_axis` is set. */
  axis: "y" | "y2" | "y3";
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
  /** Per-point overrides, bars only (charts.md § v1.1): colour, opacity, label. */
  point_colors: string[] | null;
  point_opacity: number[] | null;
  point_text: string[] | null;
  /** charts.md § v1.2 — declared by a plot that knows its figure; null = the family decides. */
  area?: boolean | null;
  end_label?: boolean | null;
  /** A flat backdrop (altitude, profile): line-strong fill, drawn first. */
  background?: boolean;
  band_opacity?: number | null;
  /** Bars only: where every bar starts (in y units), and each bar's width in x units. */
  bar_base?: number | null;
  point_widths?: number[] | null;
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

/**
 * A point pinned on the x-axis: today (dotted sun line), a race or an aid
 * station (terra dot, named), or a section boundary (thin rule, no label).
 */
export interface Marker {
  kind: "today" | "race" | "aid" | "boundary";
  x: number | string;
  label: string;
  /** A race in the current period: its label stacks above today's. */
  stacked?: boolean;
}

export type ChartFamily =
  | "tracking"
  | "comparison"
  | "function"
  | "oscillation"
  | "composition"
  | "scatter";

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
  /** Dates worth pointing at; absent from a chart cached before v1.1. */
  markers?: Marker[];
  /** "auto" (renderer's call), "closest" or "x unified". */
  hover_mode: string;
  /** Gap between bars as a share of each slot; null keeps Plotly's. */
  bargap?: number | null;
  /** charts.md § v1.2: tracking | comparison | function | oscillation | composition | scatter. */
  family?: ChartFamily | null;
  /** An oscillation's reference level; null = 0 if the data straddles it, else the mean. */
  baseline?: number | null;
  /** A binned date axis (day | week | month | quarter | year); markers align to it. */
  x_bucket?: string | null;
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
  /** The Strava athlete id — null for an account with no Strava attached. */
  id: number | null;
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
  /** Pace zones set by hand on Home, by zone key; absent without Strava. A zone
   * not in it is computed from the VMA. */
  pace_overrides?: PaceOverrides;
  /** The athlete's chosen UI language — "en" or "fr". Always set. */
  lang: string;
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
  /** Whether the signed-in account is the operator's (blog + coach). */
  is_master: boolean;
  /** The signed-in account. `email` above is the *viewed* athlete's sign-in
   * address, which differs only while a coach is viewing another athlete. */
  account: {
    id: string;
    email: string;
    role: "athlete" | "coach" | "master";
    /** Proven by a verification link or a completed password reset. */
    email_verified: boolean;
    /** Whether a verification link can be sent at all (a mail provider is set). */
    can_verify: boolean;
  };
  /** Whether a Strava athlete is attached. False: every Strava field above is
   * empty, `id` is null, and Home renders its degraded variant. */
  strava_connected: boolean;
  /** Whether Strava still answers for it — false once disconnected (history
   * kept). Absent when `strava_connected` is false. */
  strava_authorized?: boolean;
  /** The account's latest level estimate, for the Zones card's "estimated on"
   * line. Null without one, and while a coach views another athlete. */
  level_estimate: LevelEstimateMeta | null;
}

// --- Tools ------------------------------------------------------------------

export type LevelMethod = "half_cooper" | "critical_speed" | "records";

export interface LevelEstimateMeta {
  method: LevelMethod;
  created_at: string;
  vma_pace_s_per_km: number | null;
}

export interface PaceZone {
  key: string;
  low_pct: number;
  high_pct: number;
  fast_s_per_km: number;
  slow_s_per_km: number;
}

export interface LevelResult {
  method: LevelMethod;
  vma_kmh: number;
  vma_pace_s_per_km: number;
  vdot: number;
  confidence: "high" | "medium" | "low";
  extras: Record<string, number>;
  zones: PaceZone[];
  hr_max: number | null;
  hr_zones: { key: string; bpm: number }[];
  notes: string[];
  saved_at: string | null;
}

export interface ZoneDefinitions {
  vma_pace: { key: string; low_pct: number; high_pct: number }[];
  hr_max_pct: { key: "z1" | "z2" | "z3" | "z4"; pct: number }[];
  hr_pace: { key: string; low_pct: number; high_pct: number }[];
}

/** The five-level scale of the GAP and durability profiles (src/domain/assessment). */
export type AssessmentLevel = "excellent" | "good" | "average" | "limited" | "poor" | "insufficient";

export interface Assessment {
  key: string;
  /** Extra cost against the reference runner, %; `null` without data. Not shown. */
  extra_cost_pct: number | null;
  level: AssessmentLevel;
}

export interface GapSummary {
  available: boolean;
  reason?: string;
  /** Steep downhill, downhill, uphill, steep uphill — in that order. */
  terrains: Assessment[];
  /** The curve the levels were read on, against the balanced runner. */
  chart?: ChartData;
  /** When the curve was fitted, and how many runs are newer than that fit. */
  computed_at: string | null;
  new_runs: number;
}

export interface DurabilitySummary {
  /** Whether the athlete's own long runs inform the profile at all. */
  available: boolean;
  /** Long efforts, hard efforts, descents — in that order. */
  qualities: Assessment[];
  /** Projected extra cost over a long run, against the average runner. */
  chart: ChartData | null;
  /** When the model was fitted, and how many long runs are newer than that fit. */
  computed_at: string | null;
  new_runs: number;
}

export interface CoachingRequest {
  id: string;
  account_id: string;
  message: string;
  phone: string | null;
  phone_e164: string | null;
  contact: "email" | "phone";
  status: "pending" | "accepted" | "declined" | "withdrawn";
  created_at: string;
  decided_at: string | null;
}

export interface CoachingState {
  coached: boolean;
  request: CoachingRequest | null;
  can_request_again_at: string | null;
  email: string;
  proof: { coached_count: number } | null;
}

export interface CoachBoard {
  pending: (CoachingRequest & { email: string; display_name: string | null })[];
  coached: {
    account_id: string;
    email: string;
    athlete_id: number | null;
    display_name: string;
    profile_url: string | null;
    since: string;
    last_activity: string | null;
  }[];
}

export interface PageTemplate {
  key: string;
  name: string;
  description: string;
  icon: string;
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

// --- Race plan ---------------------------------------------------------------

export interface RacePlanOptions {
  /** An athlete with Strava: plans are paced on their own GAP curve. */
  signed_in: boolean;
}

export interface RacePlanAidStation {
  km: number;
  name: string;
}

export interface RacePlanParams {
  target_time_s: number;
  aid_stations: RacePlanAidStation[];
  /** Seconds after midnight; adds a time-of-day column when set. */
  start_time_s: number | null;
  /** Durability: cost drift over a long effort. Absent on plans saved before it. */
  durability?: boolean;
  temperature_start_c?: number | null;
  temperature_end_c?: number | null;
  relative_humidity_pct?: number | null;
}

export type DurabilityConfidence = "population_only" | "partially_personalized" | "personalized";

export interface RacePlanSummary {
  distance_m: number;
  elevation_gain_m: number;
  elevation_loss_m: number;
  target_time_s: number;
  gap_pace_s_per_km: number;
  average_pace_s_per_km: number;
  section_count: number;
  aid_station_count: number;
  /** Present when the plan accounts for durability. */
  durability_enabled?: boolean;
  durability_multiplier_finish?: number;
  gap_pace_finish_s_per_km?: number;
  durability_confidence?: DurabilityConfidence;
  durability_status?: "placeholder" | "fitted";
  reference_source?: string;
}

export interface RacePlanResult extends RacePlanOptions {
  /** The curve actually used — the balanced runner when a personal one fell back. */
  curve: string;
  curve_label: string;
  personalized: boolean;
  summary: RacePlanSummary;
  outputs: {
    profile: PlotOutput;
    sections: PlotOutput;
    aid_stations: PlotOutput;
    durability?: PlotOutput;
  };
  notes: string[];
}

/** A saved plan: its inputs only — the result is recomputed on every open. */
export interface SavedRacePlan {
  id: string;
  title: string;
  gpx_name: string;
  params: RacePlanParams;
  distance_m: number | null;
  elevation_gain_m: number | null;
  /** Thumbnail data; `null` when the stored GPX cannot be read. */
  preview: RacePlanPreview | null;
  /** The race's date (`YYYY-MM-DD`) and weight as an objective; `null` = not said. */
  event_date: string | null;
  importance: RacePlanImportance | null;
  /** When the stored result was computed; `null` when there is none. */
  computed_at: string | null;
  /** The plan as last computed — on save and on recompute. Only on a single plan. */
  result?: RacePlanResult | null;
  created_at: string | null;
  updated_at: string | null;
}

/** The same two levels as a diary goal's `importance`. */
export type RacePlanImportance = "primary" | "secondary";

export interface RacePlanPreview {
  /** `[latitude, longitude]`, downsampled. */
  route: [number, number][];
  /** `[km, elevation m]`, downsampled. */
  profile: [number, number][];
}

export type PaceOverrides = Record<string, { fast_s_per_km: number; slow_s_per_km: number }>;
