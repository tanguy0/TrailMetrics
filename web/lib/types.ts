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
  collapsed: boolean;
}

export interface PanelSpec {
  id: string;
  title: string;
  description: string;
  source: DataSourceSpec;
  plots: PlotSpec[];
  columns: number;
  collapsed: boolean;
}

export interface PageSpec {
  schema_version: number;
  id: string;
  name: string;
  description: string;
  icon: string;
  builtin_key: string | null;
  panels: PanelSpec[];
}

export interface PageSummary {
  id: string;
  name: string;
  description: string;
  icon: string;
  builtin_key: string | null;
  is_builtin: boolean;
  panel_count: number;
  plot_count: number;
}

// --- Parameter schema ------------------------------------------------------

export type ParamKind =
  | "bool"
  | "int"
  | "float"
  | "text"
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
}

export interface Trace {
  name: string;
  x: (number | string | null)[];
  y: (number | null)[];
  kind: TraceKind;
  color: string | null;
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

export interface ChartData {
  title: string;
  x_axis: Axis;
  y_axis: Axis;
  traces: Trace[];
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

export interface PlotOutput {
  charts: ChartData[];
  tables: TableData[];
  notes: string[];
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

export interface Athlete {
  id: number;
  firstname: string;
  lastname: string;
  display_name: string;
  profile_url: string | null;
  weight_kg: number | null;
  activity_count: number;
  sport_types: string[];
  oldest_activity: string | null;
  newest_activity: string | null;
  sync: SyncStatus;
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
