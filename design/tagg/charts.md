# Graphiques (Plotly)

Les figures sont rendues par Plotly des deux côtés (`src/domain/charts/plotly.py` et `web/components/ChartView.tsx`) à partir du même IR ; `theme.py` et `theme.ts` doivent pointer sur les tokens ci-dessous.

## Chrome

| réglage Plotly | token | note |
| --- | --- | --- |
| `paper_bgcolor`, `plot_bgcolor` | `bg-chart` (= `bg-surface`) | la figure n'a pas de cadre : la carte est le cadre |
| `font.family` | `mono` pour les ticks, `sans` pour titre et légende | `font.size` 11 pour les ticks |
| `font.color` | `ink` (titre), `chart-axis` (ticks) | |
| `xaxis.gridcolor` | aucun (`showgrid: false`) | pas de grille verticale |
| `yaxis.gridcolor` | `chart-grid` | `gridwidth` 1, `zeroline` false |
| `xaxis.linecolor`, `yaxis.linecolor` | `line` | `showline` seulement sur l'axe x |
| `hoverlabel.bgcolor` | `bg-surface` | `bordercolor` `line`, `font.color` `ink`, valeur mise en avant en `sun-ink` |
| `legend` | horizontale, au-dessus, `font.color` `ink-muted` | pas de cadre ; un trait de 18 px par entrée |
| `margin` | l 44 · r 16 · t 16 · b 32 | le titre est dans la carte HTML, pas dans la figure |

## Séries

- **L'athlète** : `chart-you-1` (forest, modèle Efficience / année courante), `chart-you-2` (terra, Auto-learning / année N-1), puis `chart-you-3..5`. Trait 2 px, jointures rondes, pas de marqueurs sauf le point actif.
- **Références** (coureur équilibré, Kilian, cibles) : `chart-ref`, trait 1,5 px, `dash: "5,4"`. Toujours derrière les séries de l'athlète.
- **Barres** (volume, distribution) : remplissage `moss` ou `forest` à 95 % pour la période courante, 28 % pour les autres ; coins `radius-sm` quand le rendu le permet, sinon droits.
- **Point actif** : cercle r 4 `bg-surface` bordé 2 px de la couleur de la série ; en survol, un second cercle r 9 à 18 % d'opacité.
- **Zones** (zones d'intensité, fitness/fatigue) : remplissage de la couleur de série à 10 %, jamais de dégradé.
- **Badge de tendance** sur la figure : `label` mono dans une pastille `forest-tint` / `terra-tint` / `sun-tint` selon le sens, en haut à droite de la zone de tracé.

## Palette multi-séries

Quand un graphique superpose plusieurs échelles de temps ou groupes : `chart-you-1` → `-2` → `-3` → `-4` → `-5` dans cet ordre (forest, terra, sun, lac, prune). Au-delà de cinq groupes, passer en petits multiples plutôt qu'en sixième couleur.
