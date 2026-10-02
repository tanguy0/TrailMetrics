import type { ReactNode, ChangeEvent } from "react";
export interface ButtonProps { variant?: "primary" | "secondary" | "ghost" | "danger" | "strava"; size?: "md" | "sm"; wide?: boolean; icon?: ReactNode; href?: string; disabled?: boolean; onClick?: () => void; children?: ReactNode; }
export interface ChipProps { tone?: "neutral" | "forest" | "terra" | "sun" | "moss" | "danger"; dot?: boolean; children?: ReactNode; }
export interface KpiTileProps { label: string; value: string; unit?: string; delta?: string; trend?: "up" | "down" | "signal"; }
export interface FieldProps { label: string; value?: string; placeholder?: string; type?: string; options?: Array<string | { value: string; label: string }>; onChange?: (e: ChangeEvent) => void; }
export interface ToggleProps { checked?: boolean; onChange?: (e: ChangeEvent) => void; children?: ReactNode; }
export interface NavRailItem { href: string; label: string; icon?: ReactNode; active?: boolean; disabled?: boolean; }
export interface NavRailProps { brand: ReactNode; homeHref?: string; switcher?: ReactNode; items: NavRailItem[]; footer?: ReactNode; }
export interface PageHeaderProps { kicker?: string; title: string; subtitle?: string; actions?: ReactNode; }
export interface PanelProps { index?: string; title: string; meta?: ReactNode; description?: string; actions?: ReactNode; children?: ReactNode; }
export interface PlotCardProps { title: string; subtitle?: string; tag?: string; span?: 4 | 5 | 6 | 7 | 8 | 12; legend?: ReactNode; children?: ReactNode; }
export interface DataTableColumn { key: string; label: string; numeric?: boolean; date?: boolean; }
export interface DataTableProps { columns: DataTableColumn[]; rows: Array<Record<string, ReactNode> & { id?: string; best?: boolean }>; }
export interface SessionCardProps { kind?: "done" | "planned" | "goal"; sport?: "run" | "bike" | "hike" | "swim" | "other"; title: string; icon?: ReactNode; stats?: string[]; tags?: ReactNode; body?: string; secondary?: boolean; }
export interface TeaserProps { kicker?: string; title: string; bullets?: string[]; actions: ReactNode; fine?: string; background?: ReactNode; }
export interface AccessItem { href: string; title: string; desc: string; icon?: ReactNode; }
export interface AccessGridProps { open: { title: string; chip?: ReactNode; items: AccessItem[] }; locked: { title: string; chip?: ReactNode; items: AccessItem[] }; }
export interface HeroStat { label: string; value: string; unit?: string; key?: boolean; }
export interface HeroProps { avatar?: string; kicker?: string; title: string; meta?: string; stats?: HeroStat[]; action?: ReactNode; }
