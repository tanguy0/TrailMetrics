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

## v1.1 — Plus de caractère

Les règles ci-dessus donnent des graphiques propres mais interchangeables. Six gestes les rendent reconnaissables sans trahir le système :

1. **Aire sous la série 1.** La série principale de l'athlète (`chart-you-1`) reçoit une aire `fillgradient` verticale de la couleur à 22 % vers 0 % (Plotly ≥ 5.20 ; sinon `fill: tozeroy` à 10 %). Une seule série par graphique a une aire ; jamais les références, jamais les comparaisons.
2. **Étiquette de fin de courbe.** Chaque série de l'athlète se termine par un marqueur r 4 et une annotation mono 11 px dans sa couleur (`4:21`, `68`), alignée à droite du dernier point. La légende reste pour les références ; pour les séries de l'athlète elle devient optionnelle quand l'étiquette de fin suffit.
3. **Survol unifié.** `hovermode: "x unified"`, ligne de repère (`spikes`) verticale `line-strong` en pointillé 2-4, infobulle carte avec la valeur de la série principale en `sun-ink` et les autres en `ink`.
4. **Axes plus légers.** `nticks: 5` en y, pas de ligne d'axe y, grille `chart-grid` ; en x une seule ligne `line` et les ticks sans trait. Les titres d'axe disparaissent quand l'unité est dans le sous-titre de la carte.
5. **Barres sémantiques.** `marker.cornerradius: 4`. Dans la *carte des pentes* et la *distribution*, les barres prennent le sens de la donnée : descentes `moss`, plat `forest`, montées `terra` (le signe de la pente, pas l'ordre des séries). Dans *volume*, période courante 95 %, autres 28 % — inchangé.
6. **Repères temporels.** « Aujourd'hui » = ligne verticale `sun` pointillée avec une étiquette mono ; une course = marqueur `terra` au-dessus de l'axe avec son nom. Les bandes (zones, blocs d'entraînement) restent à 10 % et perdent leur bordure.

Deux conséquences : la **sparkline** (`tm-kpi__spark`) est une mini-version de ces règles — trait 1,6 px, aire 12 %, point de fin, rien d'autre ; et *fitness / fatigue / forme* se dessine fitness en `forest` avec aire, fatigue en `terra` fin sans aire, forme en `sun` en barres 60 % positive/négative autour de zéro.

## v1.2 — Quelle aire, quel trait, pour quel graphique

L'aire sous « la série 1 » appliquée partout produit des figures étranges : une courbe GAP remplie jusqu'au bas, une distance cumulée où une seule des trois années est pleine. L'aire est un **signe**, pas un ornement : elle dit « ceci est une quantité qui s'accumule dans le temps et que vous suivez seule ». Elle n'a donc de sens que dans un cas, et le reste des graphiques se classe en cinq familles avec chacune ses règles.

| famille | reconnaître | aire | traits | étiquettes de fin | légende |
| --- | --- | --- | --- | --- | --- |
| **Suivi** — une série de l'athlète dans le temps (volume, fitness, poids, VMA) | axe x date, 1 série athlète, zéro proche des données | **oui**, sur cette série | 2,2 px | oui | non (l'étiquette suffit) ; références seules |
| **Comparaison** — plusieurs périodes ou groupes (2025 vs 2026, blocs, distance cumulée par an) | axe x date ou km, ≥ 2 séries athlète sur le même axe | **non** | courante 2,4 px ; autres 1,5 px à 70 % d'opacité ; au-delà de 3 autres, 55 % | sur la courante seulement | oui, toutes |
| **Fonction** — une courbe qui n'est pas dans le temps (coût vs pente, allure vs FC, zones) | axe x numérique non temporel | **jamais** | athlète 2,2 px, références 1,5 px pointillé | non ; le survol unifié fait le travail | oui |
| **Oscillation** — un ratio ou un écart autour d'un niveau (puissance/FC, forme, écart à la référence) | zéro loin des données, ou valeurs de part et d'autre d'une ligne de base | **non** ; une ligne de base `line-strong` et, pour forme, les barres ambre | 2 px | oui | selon le nombre de séries |
| **Composition** — aires empilées, barres (carte des pentes, distribution, volume par sport) | `stackgroup` ou barres | l'empilement est l'aire ; pas de dégradé | hairline 0,35 px entre bandes | non | oui |
| **Nuage** — points (séance par séance) | scatter | non | marqueurs r 4 à 60 % d'opacité, tendance en forest 2 px | non | oui si ≥ 2 groupes |

Deux règles transversales :

- **L'agrégation décide.** Somme ou compte (volume, D+, nombre, cumul) → *Suivi* : part de zéro, aire. Moyenne, max ou min (puissance/FC, allure, FC, poids, VMA) → *Oscillation* : pas d'aire, ligne de base à la moyenne de la fenêtre, étiquette de fin. Le zéro « proche » (règle `_ZERO_REACH` : à moins d'une demi-étendue sous le minimum) ne sert plus qu'en secours, quand l'agrégation est inconnue. Un axe x qui n'est pas une date mais reste temporel (`x_mode = elapsed`) se déclare : l'IR ne peut pas le deviner.
- **Trois choses s'appellent « aire », une seule compte.** L'aire en dégradé est le signe du *Suivi* et de la fatigue déclarée : une par figure. Les **bandes** (IQR, incertitude) et les **fonds** (`background` : altitude, profil) sont à plat, sans dégradé, et ne comptent pas. Une figure peut avoir un fond, une bande et une aire.
- **Le double axe annule l'aire** et les étiquettes de fin : deux unités, deux couleurs d'axe, rien d'autre.

### Repères sur un axe en paquets

Quand l'axe x est en semaines ou en mois, « aujourd'hui » **s'aligne sur le dernier point de données** (le centre du paquet courant) et non sur la date réelle — sinon la ligne flotte entre le dernier point et le bord. Le marqueur course garde sa date exacte, lui ; s'il tombe dans le paquet courant, il se dessine sur le point du paquet et son étiquette passe au-dessus de celle d'aujourd'hui. Sur un axe journalier, les deux restent exacts. Quand aujourd'hui coïncide avec le dernier point, l'étiquette de fin reste à droite du point et l'étiquette « aujourd'hui » au-dessus : elles ne se chevauchent pas.

## v1.2 — Cas particuliers (priment sur les familles)

Un plot qui sait ce qu'il montre déclare sa mise en forme (`family` et les options ci-dessous dans l'IR) ; le classement automatique ne sert que pour `metric_trend` générique.

**Fitness · fatigue · forme** (`fitness_fatigue`). La donnée qu'on regarde est la **fatigue** : c'est elle qui porte l'aire (`chart-you-2` terra, dégradé 22 % → 0), 2,2 px, étiquette de fin. La fitness devient un trait fin `chart-you-1` 1,5 px sans aire, étiquette de fin. La forme reste en barres `sun` autour de zéro (0,75 positive / 0,4 négative), ligne de base `line-strong`. Survol unifié. C'est la seule figure où l'aire n'est pas sur la série 1 — parce que la question posée est « suis-je fatigué », pas « suis-je en forme ».

**Progression des records** (`records`, step, axe inversé). Pas d'aire (axe inversé). Chaque record est un palier : marqueur r 4 sur la marche uniquement (pas sur chaque point), le record courant en étiquette de fin, et **le record de moins de 30 jours en marqueur `sun` r 6** — la même règle que la pastille « nouveau ». Plusieurs distances ensemble = comparaison : la distance choisie en premier 2,4 px, les autres 1,5 px à 70 %.

**Durabilité** (`durability_curve`). La **bande interquartile des observations est l'aire** de la figure (`chart-you-1` à 10 %), la médiane en marqueurs r 3 sans trait ; le modèle personnalisé en trait 2,2 px `chart-you-1`, le prior population en `chart-ref` pointillé 5-4. Famille *fonction* : pas d'étiquette de fin. Sur la projection, la bande du modèle personnalisé remplace celle des observations — jamais deux bandes.

**Évolution dans une activité** (`stream_evolution`). L'altitude, quand elle est sélectionnée, est **toujours dessinée en premier et en fond** : aire `line-strong` à 35 %, sans trait, sur l'axe droit, hors légende sauf si c'est le seul signal. Par-dessus, les signaux : allure (GAP et brute) sur un **axe inversé** (plus vite = plus haut), FC en `chart-you-2` 1,5 px, puissance en `chart-you-3`. Plusieurs activités superposées = comparaison : l'activité la plus récente 2,4 px, les autres 1,5 px à 70 % ; au-delà de quatre, l'app demande d'en retirer plutôt que de passer sous 55 %. Survol unifié par distance.

**Carte des pentes** (`gradient_map`, 100 % empilé). Ordre de l'empilement fixe : descentes en bas (`moss`), plat au milieu (`forest`), montées en haut (`terra`) ; **à l'intérieur d'une famille, l'opacité croît avec la pente** (−5 % à 0,5, −15 % à 0,75, −25 % à 1 ; idem côté montée). Légende dans le même ordre que l'empilement, de bas en haut. Hairline 0,35 px entre bandes.

**Distribution** (`metric_distribution`). Un groupe = barres (`cornerradius` 4). **Deux groupes ou plus = contours** : chaque groupe en `STEP` 2 px sans remplissage, pour que les histogrammes se superposent sans se masquer. L'histogramme de pente garde ses couleurs par signe et ses trois pics étiquetés.

**Tendance générique** (`metric_trend`). Trois cas qui échappent au classement automatique : `split_by_sport` colore par **sport** (`sport-run`, `sport-bike`, `sport-hike`, `sport-swim`) et non par le cycle `chart-you` ; `x_mode = elapsed` (superposition de saisons) est toujours une *comparaison*, la saison la plus récente étant la courante ; `cumulative` en `STEP` ne reçoit une aire que seul (tracking), jamais à plusieurs. Une métrique d'allure met son axe à l'envers.

**Ressenti hebdomadaire** (`weekly_feel`). Le RPE prend la **couleur de sa pastille** : barres `moss` ≤ 4, `sun` 5–7, `danger` ≥ 8, chacune à 0,85 ; le ressenti en trait `moss-ink` 1,5 px avec marqueurs r 3 sur l'axe droit (échelle fixe, pas d'autorange). Pas d'aire, pas d'étiquette de fin : l'échelle est ordinale.

**Nuage** (`metric_scatter`). Points r 4 à 0,6, un groupe = `chart-you-1`, groupes suivants dans le cycle ; la tendance en trait 2 px de la couleur du groupe, pas en forest si le groupe est terra. Axe d'allure inversé.

**Plan de course** (profil, sections, ravitos). Le profil altimétrique en aire `line-strong` 35 % ; les **ravitos en marqueurs `terra`** sur l'axe x — le même dessin que le marqueur course — avec leur nom ; les limites de section en traits verticaux `line` ; la section survolée se colore `forest-tint`. Le graphique des allures par section en barres, chacune colorée par son écart à l'allure moyenne : plus lente `terra`, plus rapide `moss`, à ±3 % près `forest`.

**Axes d'allure, partout.** Inversés (plus vite = plus haut), ticks `m:ss`, et aucune aire ne s'y pose.
