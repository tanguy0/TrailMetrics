# Brief Claude Code — Refonte TAGG (ex-TrailMetrics), une seule PR

Ce dossier `design/tagg/` est la source de vérité du nouveau design. Il a été produit avec Claude (direction « Plateforme », logo T2) et validé par Tanguy. Ne pas réinventer : appliquer.

## Ce que contient `design/tagg/`

| fichier | rôle |
| --- | --- |
| `README.md` | le brand book : principes, rôles des couleurs, typo, rayons, ombres, icônes — **lire en premier** |
| `visitor.md` | l'expérience du visiteur sans compte (modèle freemium) — **lire en deuxième** |
| `tokens.json` | la source de vérité des tokens (couleurs, type, espacements, rayons, ombres, avec un `usage` par token) |
| `tokens.css` | le même, compilé en variables CSS `:root` + classes de texte — remplace le bloc `:root` de `web/app/globals.css` |
| `theme_tokens.py` | les mêmes couleurs en constantes Python + le mapping des rôles Plotly — remplace `src/domain/gap/theme.py` |
| `charts.md` | réglage par réglage ce que Plotly doit faire (`plotting_common.py`, `charts/plotly.py`, `ChartView.tsx`) |
| `migration.md` | la table ancienne variable / ancienne classe → nouvelle |
| `components/components.css` | les 13 composants en classes `tm-*` (prêt à coller) |
| `components/components.js`, `index.d.ts` | les mêmes en React (optionnel ; l'app est en CSS écrit main, les classes suffisent) |
| `components/*.md` | guidelines par composant : quand l'utiliser, quelles variantes, ce que le consommateur fournit |
| `logo/*.svg` | lockup, lockup sur rail, mot-symbole ×2, tuile ×2, favicon |

## Une seule PR, en commits ordonnés

La PR `feat(design): refonte TAGG` fait tout. Elle est découpée en **commits atomiques dans l'ordre ci-dessous** pour rester relisible et bisectable, mais rien n'est mergé entre deux étapes : pas d'alias de compatibilité, pas d'état intermédiaire à maintenir. À la fin de chaque commit, `npm run typecheck` et `npm run lint` passent et l'app démarre.

### Commit 1 — Tokens et polices
- Copier `design/tagg/tokens.css` → `web/app/tokens.css`, `design/tagg/components/components.css` → `web/app/components.css` ; les importer depuis `layout.tsx` avant `globals.css`.
- Dans `globals.css`, **supprimer** le bloc `:root` (et tout `--scale-*`, `--sport-scale-*`, `--tone-*`, `.scale-N`) ; réécrire chaque usage avec la table de `migration.md`. `web/lib/colorScale.ts` est supprimé.
- `<link>` Google Fonts (Manrope + DM Mono, cf. README) dans `layout.tsx` ; `body{font-family:var(--font-sans)}` ; `font-variant-numeric: tabular-nums` sur les classes mono.
- `src/domain/gap/theme.py` ← `design/tagg/theme_tokens.py` (les noms exportés existants sont tous définis) ; `web/lib/theme.ts` lit les mêmes valeurs.

### Commit 2 — Nom, logo, chrome
- Renommer « TrailMetrics » → « TAGG » partout (`translations.py`, `layout.tsx` title/meta, `package.json`, README, `Sidebar` alt). Motto « Train · Analyse · Guide · Grow » en `kicker` sous le lockup sur la landing et dans la signature du blog uniquement.
- `public/logo.webp`, `favicon.ico`, `background.webp` → les SVG de `design/tagg/logo/` (le background n'a plus d'usage : supprimer).
- `Sidebar.tsx` → `.tm-rail` (cf. `components/NavRail.md`) : SVG inline au trait à la place des emojis, logo `tagg-lockup-on-rail.svg` à 28 px, sélecteur coach/athlète en `.tm-rail__switcher`.
- `.page-header` → `.tm-page-header` ; `.button*` → `.tm-btn*` ; `.tag` / `.card-badge` / `.session-tag` / `.trend-badge` → `.tm-chip*` ; `input/select/textarea` → `.tm-input` / `.tm-select` ; cases binaires → `.tm-toggle` ; `.modal-panel` → `.tm-modal`.

### Commit 3 — Analyses et graphiques
- `.panel` → `.tm-panel` avec numéro de panneau (`.tm-panel__index` = `01`, `02`…) et source de données en pastilles dans `.tm-panel__meta` ; `.plot-card` / `.plot-grid` → `.tm-plot` / `.tm-plot-grid` (12 colonnes, `tm-plot--N`).
- Appliquer `charts.md` dans `plotting_common.py`, `charts/plotly.py`, `ChartView.tsx` : fond `bg-chart`, grille horizontale seule, axes mono 11 px, séries `CHART_YOU_1..5`, références `CHART_REF` en pointillé (5-4 et 2-4), infobulle carte, légende au-dessus sans cadre, marges l44 r16 t16 b32, titre retiré de la figure (il est dans la carte).
- `.tile` / `.metric` → `.tm-kpi` ; `.table*` → `.tm-table` (numériques `is-num`, dates `is-date`, `.cell--best` → `tr.is-best`).

### Commit 4 — Entraînement
- `TrainingScreen.tsx` : `.training-session--*` → `.tm-session[data-sport]`, `.training-pill--*` → `.tm-session--planned` / `--goal` (`is-secondary` pour l'importance secondaire), `.training-day--today` → `.tm-day--today`, `.week-summary` → `.tm-week-summary`.
- Pastilles RPE : `moss` ≤ 4, `sun` 5–7, `danger` ≥ 8 (remplace la fonction verte→rouge).

### Commit 5 — Expérience visiteur (freemium)
Suivre `design/tagg/visitor.md` à la lettre ; composants `Teaser.md` et `AccessGrid.md`.
- `app/page.tsx` (landing) : lockup + motto, une phrase, puis `.tm-access` (colonne **Sans compte** : Plan de course, Blog ; colonne **Avec Strava** : Accueil, Analyses, Entraînement), puis le bouton Strava et sa phrase de confiance. Supprimer l'ancienne `feature-list`.
- `Sidebar.tsx` visiteur : deux groupes `.tm-rail__group` (« Ouvert », « Avec Strava »), items verrouillés cliquables avec cadenas (`.tm-rail__link--locked`), bouton Strava `sm` plein largeur en pied de rail. Plus d'items inertes à 45 %.
- `app/home/page.tsx`, `app/pages/page.tsx`, `app/training/page.tsx` : remplacer `if (!(await readSession())) redirect("/")` par le rendu d'un `Teaser` dans le shell normal (rail compris). L'arrière-plan du teaser est la **structure vide de la page** (tuiles, grille de graphiques) rendue avec les vrais composants, pas une image. Bénéfices (3 max) et textes : voir `visitor.md`. Pas de « Pro / Premium / Débloquer » : le niveau s'appelle « Gratuit · avec Strava ».
- Plan de course et Blog : une seule ligne `body-sm` `ink-muted` en fin de page avec un lien texte vers la connexion (cf. `visitor.md`). Pas de bandeau.
- Hors périmètre de cette PR mais à préparer : l'athlète de démonstration (« Voir un exemple »). Laisser le bouton fantôme hors du DOM tant que le dataset n'existe pas ; ne pas le simuler.

### Commit 6 — Nettoyage
- Retirer de `globals.css` toute règle devenue morte (objectif : bien sous les 3 290 lignes), les classes renommées, les assets orphelins. `grep -rn "TrailMetrics\|--primary\|--sunrise\|--terracotta\|scale-" web src` doit ne rien retourner.

## Règles pendant tout le chantier
- **Ne jamais inventer une couleur** : tout hex vient de `tokens.json`. Besoin non couvert → ajouter le token dans `tokens.json` + `tokens.css` + `theme_tokens.py`, avec son `usage`, dans le même commit.
- Un seul bouton primaire visible par écran ; `terra` = « vous » uniquement ; `moss` et `sun` jamais en texte.
- Pas d'emoji, pas de dégradé, pas de bord gauche coloré sauf `.tm-session`.
- Chiffres toujours en mono, alignés à droite dans un tableau.
- `ink-faint` seulement dans une carte blanche, pas sur `bg-page` sous 14 px.
- Garder `theme.py` et `theme.ts` synchronisés avec `tokens.json` (ajouter un test qui compare les trois).
- Les pages verrouillées ne redirigent plus : elles rendent un `Teaser`.

## Prompt à coller dans Claude Code (à la racine du repo)

> Lis `design/tagg/README.md`, `design/tagg/visitor.md`, `design/tagg/migration.md` et `design/CLAUDE_CODE_BRIEF.md`. Nous refondons le design et l'expérience visiteur de l'app selon ce système, en **une seule PR** découpée en 6 commits dans l'ordre du brief. Avant de coder, donne-moi le plan de fichiers touchés par commit. Puis exécute commit par commit : à chaque fin de commit, `npm run typecheck` et `npm run lint` passent, et tu me montres un résumé. Aucune couleur hors de `tokens.json`. Quand une règle est ambiguë pour un composant, cite la ligne du README, de `visitor.md` ou de `components/*.md` concernée avant de trancher.
