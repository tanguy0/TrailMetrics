# Migration depuis globals.css

L'app actuelle est en CSS écrit main (3 290 lignes) avec les variables ci-dessous ; voici où chacune va. `tokens.css` (généré par ce système) remplace le bloc `:root` de `globals.css` ; `components/components.css` remplace les règles des composants listés.

## Variables

| ancienne | nouvelle | remarque |
| --- | --- | --- |
| `--bg` #fbf8f3 | `--bg-page` #f4f1ea | grisé d'un cran |
| `--surface` #fffdf9 | `--bg-surface` #ffffff | blanc pur |
| `--surface-alt` #f0eadf | `--bg-surface-alt` #f8f6f1 | |
| `--grid` #cfc3ae | `--line` #e8e2d6 / `--chart-grid` #ece7dd | la grille Plotly et les filets UI se séparent |
| `--spine` #b8ac97 | `--line-strong` #d5cdbe | |
| `--text`, `--muted` | `--ink`, `--ink-muted` | identiques |
| `--primary` `--terracotta` `--sunrise` `--moss` `--danger` | `--forest` `--terra` `--sun` `--moss` `--danger` | valeurs inchangées |
| `--tone-*-tint`, `--sunrise-tint`, `--moss-tint`, `--danger-tint` | `--forest-tint` `--terra-tint` `--sun-tint` `--moss-tint` `--danger-tint` | |
| `--tone-blue`, `--tone-plum` | `--sport-bike`, `--sport-swim` (et `--chart-you-4/5`) | |
| `--scale-1..6`, `--sport-scale-1..4`, `.scale-N` | supprimés | un panneau est numéroté (`.tm-panel__index`), un sport a sa couleur fixe `--sport-*` |
| `--radius` 10px | `--radius-md` | plus `-sm` `-lg` `-xl` `-pill` |
| `--tile-height` 6.6rem | conservé tel quel | hors système |

## Classes

| ancienne | nouvelle |
| --- | --- |
| `.button`, `--ghost`, `--danger`, `--small`, `--wide`, `--strava` | `.tm-btn`, `.tm-btn--secondary`, `--ghost`, `--danger`, `--sm`, `--wide`, `--strava` |
| `.tag`, `.card-badge`, `.session-tag`, `.trend-badge` | `.tm-chip` + `--forest` `--terra` `--sun` `--moss` `--danger`, `.tm-chip--dot` |
| `.sidebar`, `.sidebar__link`, `.coach-switcher` | `.tm-rail`, `.tm-rail__link`, `.tm-rail__switcher` |
| `.page-header` | `.tm-page-header` (surtitre `kicker` + titre `display-lg` + actions) |
| `.panel`, `.panel__header`, `.panel__title` | `.tm-panel`, `.tm-panel__head`, `.tm-panel__index`, `.tm-panel__title`, `.tm-panel__meta` |
| `.plot-card`, `.plot-grid` | `.tm-plot`, `.tm-plot-grid` (12 colonnes, `--span-N`) |
| `.tile`, `.tile-grid`, `.metric` | `.tm-kpi`, grille flex |
| `.table`, `.table-block`, `.cell--best` | `.tm-table`, `.tm-table td.is-num`, `.tm-table tr.is-best` |
| `.training-session--*`, `.training-pill--*`, `.training-day--today` | `.tm-session` + `data-sport`, `.tm-session--planned`, `.tm-session--goal`, `.tm-day--today` |
| `.week-summary` | `.tm-week-summary` |
| `input`, `select`, `textarea`, `.param` | `.tm-field`, `.tm-field__label`, `.tm-input`, `.tm-select`, `.tm-toggle` |
| `.modal-panel` | `.tm-modal` (`radius-xl`, `shadow-pop`) |

## Python (`theme.py`)

`FIGURE_FACE` et `AXES_FACE` → `bg-chart` (#ffffff) ; `GRID` → `chart-grid` ; `SPINE` → `line` ; `TEXT` → `ink` ; `TIME_SCALE_CYCLE` → `chart-you-1..5` ; `BALANCED_RUNNER` et `KILIAN` → `chart-ref` (ils ne se distinguent plus par la couleur mais par le libellé et un `dash` différent : 5-4 et 2-4) ; `LOW_INTENSITY`/`HIGH_INTENSITY` → `moss` / `forest-hover`.
