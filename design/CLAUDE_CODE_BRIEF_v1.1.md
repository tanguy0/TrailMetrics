# Brief Claude Code — TAGG v1.1 : contraste, relief, graphiques (une PR)

La v1.0 est en place et bonne. Cette PR corrige **un** défaut — tout est au même poids, les accents ont disparu — et donne du caractère aux graphiques. Ce n'est pas une refonte : ne pas toucher à la structure des pages, aux composants qui marchent, ni au rail.

## Fichiers mis à jour dans `design/tagg/` (remplacer les existants)
`tokens.json` (v2), `tokens.css`, `theme_tokens.py`, `README.md`, `charts.md` (nouvelle section *v1.1*), `components/components.css`, `components/components.js`, `components/index.d.ts`, et deux nouveaux : **`contrast.md`** (lire en premier), `components/Hero.md`, `components/Highlights.md`.

## Commits, dans l'ordre

### 1 — Tokens
- `web/app/tokens.css` ← `design/tagg/tokens.css`. Changement visible : `--bg-page` passe à #eee9de. Nouveaux : `--bg-hero`, `--on-hero-muted`, `--bg-tile`, `--hl`.
- `src/domain/gap/theme.py` ← `theme_tokens.py` (nouvelles constantes `SLOPE_*`, `FITNESS/FATIGUE/FORM`, `AREA_ALPHA_TOP`, `TODAY_MARKER`, `RACE_MARKER`). `web/lib/theme.ts` suit.
- `web/app/components.css` ← `components/components.css` (ajoute `.tm-hero*`, `.tm-kpi--flat`, `.tm-kpi--forest/--terra/--sun/--moss`, `.tm-kpi__spark`, `.tm-section*`, `.tm-panel--tint`, `.tm-callout*`, `.tm-hl`, `.tm-num*`).

### 2 — Tuiles plates et surtitres (toutes les pages)
- Toute `.tm-kpi` rendue **à l'intérieur** d'une `.card-block`, `.tm-panel` ou `.tm-modal` prend `tm-kpi--flat` (fond crème, sans bord, sans ombre). Les tuiles posées directement sur la page ne changent pas. Supprimer la règle `.card-block .tm-kpi{box-shadow:none}` devenue inutile.
- `.card-block__title` → `.tm-section` avec `tm-section__kicker`. Chaque carte reçoit un surtitre (contexte temporel ou « Vous ») et un rôle : Accueil — Historique forest, Santé terra, Forme sun, Ressenti moss, Zones terra ; Entraînement — bilan de semaine forest ; Analyses — les panneaux gardent leur numéro, pas de kicker.

### 3 — Accueil : la vedette et les têtes d'affiche (cf. `contrast.md` §1-3 et la maquette *Accueil · v1.1* du canvas)
- `.hero` → `tm-hero` : avatar, kicker « Semaine N · dates », nom, meta (synchro, nb d'activités), **trois chiffres de la semaine courante** — volume, D+, forme — dont **forme** en `is-key` (sun). Action « Importer » en secondaire ; le bouton primaire de la page n'est pas dans le hero. Supprimer le bord sun de l'avatar (`.hero__avatar{border:3px solid var(--sun)}`) : la couleur est dans le chiffre, pas dans un cadre.
- Une tuile tête d'affiche par carte (deux max), chiffre coloré + sparkline 13 semaines : Historique → distance totale `tm-kpi--forest` ; Santé → poids `tm-kpi--terra` ; Forme → forme `tm-kpi--sun` (fitness et fatigue restent en ink) ; Ressenti → ressenti `tm-kpi--moss`. La sparkline est un `<svg>` inline (trait 1,6 px, aire 12 %, point de fin), données déjà présentes dans `HomeSummary` ou via le panneau `metric_trend` existant.
- Records : le plus récent en pastille `sun` avec « nouveau », les autres neutres.
- Les `.note` d'information → `tm-callout` (sun), d'erreur → `tm-callout--terra`. Au plus une par carte.

### 4 — Entraînement et Analyses
- Entraînement : le bilan de la **semaine courante** passe en `tm-hero` compact (sans avatar, stats en colonne) ; les autres semaines gardent le bilan crème. Un seul hero sur l'écran.
- Analyses : pas de hero. `tm-panel--tint` autorisé pour **un** panneau : celui marqué comme principal dans la page (le premier par défaut). Les descriptions de panneau peuvent porter un `tm-hl` sur le groupe de mots clé.
- Landing visiteur : la colonne « Avec Strava » de l'`AccessGrid` prend `tm-panel--tint`.

### 5 — Graphiques (cf. `charts.md` §v1.1, maquette *Graphiques · v1.1* du canvas)
Dans `plotting_common.py`, `charts/plotly.py`, `ChartView.tsx` — les deux implémentations restent synchrones :
1. Série 1 de l'athlète : `fill: "tozeroy"` avec `fillgradient: {type:"vertical", colorscale:[[0, rgba(col,0)],[1, rgba(col,0.22)]]}` (Plotly ≥ 5.20 ; vérifier la version du bundle `plotly.js` côté web, sinon `fillcolor` à 10 %). Une seule aire par figure, jamais sur les références ni les comparaisons.
2. Étiquette de fin : pour chaque série de l'athlète, un marqueur r 4 sur le dernier point + `annotation` mono 11 px dans la couleur de la série, `xanchor:"left"`, `xshift: 9`. La légende des séries de l'athlète devient `showlegend: false` quand l'étiquette de fin est présente ; les références gardent la légende.
3. `hovermode: "x unified"`, `xaxis.showspikes: true`, `spikecolor: LINE_STRONG`, `spikedash: "2px,4px"`, `spikethickness: 1`, `spikemode: "across"`. Infobulle : série principale en `SUN_INK` gras, autres en `INK`.
4. Axes : `yaxis.nticks: 5`, `yaxis.showline: false`, `xaxis.ticks: ""`, `xaxis.showline: true` en `LINE`. Retirer `yaxis.title` quand la carte HTML porte déjà l'unité dans son sous-titre (le `PanelSpec` sait l'unité).
5. Barres : `marker.cornerradius: 4`. Carte des pentes et distribution : couleur par signe de la pente — `SLOPE_DOWN` / `SLOPE_FLAT` / `SLOPE_UP` — opacité 1 sur les trois tranches les plus fréquentes, 0,55 ailleurs, valeur en étiquette au-dessus de ces trois-là. Volume : inchangé (courante 95 %, autres 28 %).
6. Repères : « aujourd'hui » = `shape` ligne verticale `TODAY_MARKER` pointillée 2-4 + annotation mono au-dessus ; course = marqueur `RACE_MARKER` au-dessus de l'axe x avec son nom. Bandes à 10 % sans bordure.
- Fitness / fatigue / forme : fitness `FITNESS` 2,2 px avec aire ; fatigue `FATIGUE` 1,5 px sans aire ; forme `FORM` en barres 60 % autour de zéro (opacité 0,75 positive, 0,4 négative).

### 6 — Nettoyage
`grep -rn "hero__avatar\|card-block__title\|\.note\b" web` ne retourne plus de règles actives ; `tokens.css`/`theme.py`/`theme.ts` synchrones (le test existant).

## Garde-fous
- **Une vedette par page**, jamais deux `tm-hero` ou un hero + un `tm-panel--tint` sur le même écran.
- **Une ou deux tuiles colorées par carte**, et la section et sa tuile partagent le même rôle.
- `moss` et `sun` toujours via `-ink` en texte. `tm-hl` : un par paragraphe, groupe de mots, pas phrase entière.
- Aucune couleur hors `tokens.json`. Les rôles ne changent pas : forest structure, terra « vous », sun signal, moss positif.

## Prompt à coller
> Lis `design/tagg/contrast.md`, la section *v1.1* de `design/tagg/charts.md`, `design/tagg/components/Hero.md` et `Highlights.md`, puis `design/CLAUDE_CODE_BRIEF_v1.1.md`. Applique la v1.1 en **une PR**, 6 commits dans l'ordre du brief. Donne-moi d'abord le plan de fichiers par commit. `npm run typecheck` et `npm run lint` au vert à chaque commit. Respecte les garde-fous : une vedette par page, une à deux tuiles colorées par carte, aucune couleur hors `tokens.json`.
