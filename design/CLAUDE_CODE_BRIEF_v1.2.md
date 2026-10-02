# Brief Claude Code — TAGG v1.2 : graphiques par famille, repères, densité, hero course (une PR)

Fichiers à **fusionner** (jamais remplacer) dans `design/tagg/` : `charts.md` (nouvelle section *v1.2*), `README.md`, `components/Hero.md`, `components/KpiTile.md`, nouveau `density.md`, et `components/components.v1.2.additions.css` à **ajouter à la fin** de `web/app/components.css` (puis supprimer le fichier additions).

## Commits

### 1 — Familles de graphiques (`charts.md` § v1.2)
Remplacer `_area_trace` (« la série 1 a l'aire ») par une **classification** de la figure, calculée une fois dans `plotly.py` et miroir dans `ChartView.tsx` :
- `family = "tracking"` : axe x date, exactement 1 série athlète sur l'axe gauche, pas de `y2_axis`, pas de `stack_group`, zéro « proche » (règle `_ZERO_REACH` existante). → aire + étiquette de fin, légende masquée pour la série athlète.
- `"comparison"` : ≥ 2 séries athlète sur le même axe (années, groupes, distance cumulée `STEP`). → **pas d'aire** ; la série courante (la dernière période, ou `chart-you-1`) 2,4 px ; les autres 1,5 px opacité 0,7 (0,55 au-delà de 3) ; étiquette de fin sur la courante seulement ; légende complète.
- `"function"` : axe x non temporel (pente, FC, km dans une course). → jamais d'aire ni d'étiquette de fin ; survol unifié.
- `"oscillation"` : zéro loin des données ou valeurs de part et d'autre d'une base (puissance/FC, forme, écarts). → pas d'aire ; ligne de base `LINE_STRONG` à la valeur de référence (1,0 ; 0 ; la moyenne) ; étiquette de fin oui.
- `"composition"` : `stack_group` ou barres → inchangé (l'empilement est l'aire).
- `"scatter"` → marqueurs opacité 0,6, tendance forest.
- Tout `y2_axis` → ni aire ni étiquettes.
L'IR peut porter un `family` optionnel pour qu'un plot force sa famille ; sinon elle est déduite. Puis appliquer la section **Cas particuliers** de `charts.md` plot par plot : chaque plot nommé déclare sa `family` et ses options dans l'IR (nouveaux champs optionnels sur `Trace` : `area: bool`, `end_label: bool`, `background: bool` ; sur `ChartData` : `family`). En particulier `fitness_fatigue` met l'aire sur la **fatigue**, pas la fitness. Vérifier sur : *Distance cumulée* (comparison), *Courbes GAP* (function), *Volume hebdo* seul (tracking), *Puissance/FC* (oscillation), *Fitness/fatigue/forme* (oscillation ; la **fatigue** porte l'aire, cas particulier de charts.md).

### 2 — Repères sur axes en paquets
Dans `plotting_common.py` et `ChartView.tsx` : quand l'axe x est en semaines/mois (le plot le sait : `bucket`), le marqueur `today` prend **x = x du dernier point** de la série principale, pas la date réelle. Marqueur `race` dans le paquet courant : même x, étiquette au-dessus de celle d'aujourd'hui (`yshift` +14). Axe journalier : inchangé. Sur l'Accueil, les quatre graphiques doivent avoir « aujourd'hui » exactement sur le dernier point.

### 3 — Hero de Plan de course (`Hero.md` § Plan de course)
`RacePlanHero` : `kicker` = « Plan de course · courbe {curve_label}{ · personnalisée} » ; `title` = nom de la course ; nouveau `meta` = stratégie en une ligne composée depuis `summary` : « GAP {gap_pace_s_per_km} /km · réel {average_pace_s_per_km} /km · {section_count} sections · {aid_station_count} ravitos · dérive ×{durability_multiplier_finish} à l'arrivée ({durability_confidence}) », chaque fragment omis si absent. Stats : Temps visé (`is-key`), Distance, D+ / D− (`+2 100 / −1 980 m`), Allure GAP. Le `<h2>` « Temps visé » disparaît. Les chaînes passent par `translations.py` (fr + en).

### 4 — Densité (`density.md`)
- `format.ts` : `formatDate(value, style = "short" | "relative" | "long")`. `short` = `30 sept.` (+ ` 25` si année ≠ courante) ; `relative` = `il y a 12 min` / `hier` / jour de semaine / puis `short` ; `long` = `mardi 30 septembre 2026`. Locale depuis les strings (fr/en). Plus aucun `toISOString().slice(0,10)` dans l'UI ; `grep -rn "slice(0, 10)" web` ne retourne que du code non-UI.
- Hero/meta/synchro → `relative` ; tuiles, cellules, pastilles, records → `short` ; titres de page/PDF → `long`.
- Intervalles d'allure : `${fast}–${slow}` sans espaces, `nowrap`, et l'unité `/km` remonte dans le libellé (« Allure (/km) ») quand la valeur est un intervalle. Colonnes numériques `min-width: 9ch` ; un tableau trop large défile dans `.tm-table-scroll`.
- Tuile : la valeur reçoit `tm-kpi__num--lg` (> 5 caractères) ou `--sm` (> 9), unité exclue du compte. Dans `HomeScreen`, les tuiles dont le footnote est une date passent en `short`.
- Durées : `formatHms` sans zéro de tête (`1:12`, pas `01:12:00`), `1 062 h` au-delà d'un jour.
- `nowrap` + `tabular-nums` sur toutes les valeurs (CSS fourni). Noms d'activité en cellule : `is-text` (ellipsis + `title`).

### 5 — Nettoyage
`grep -rn "Temps visé\|target\." web/components/RacePlanScreen.tsx` : un seul affichage du temps visé dans le hero. Les deux moteurs de graphiques produisent la même famille pour chaque chart du jeu de test (ajouter un test de parité sur `family`).

## Garde-fous inchangés
Une aire au plus par figure et uniquement en famille *tracking* ; aucune couleur hors `tokens.json` ; valeurs jamais sur deux lignes.

## Prompt
> Lis la section *v1.2* de `design/tagg/charts.md`, `design/tagg/density.md`, la fin de `design/tagg/components/Hero.md`, puis `design/CLAUDE_CODE_BRIEF_v1.2.md`. Applique la v1.2 en une PR, 5 commits dans l'ordre. Plan de fichiers par commit d'abord. Pour le commit 1, montre-moi la famille déduite pour chaque chart du jeu de test avant de changer le rendu.
