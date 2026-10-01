## Design — TAGG

- Le design de l'app est défini par `design/tagg/` (source de vérité : `tokens.json` ; brand book : `README.md`). Toute décision visuelle s'y réfère ; en cas de doute, lire `design/tagg/components/<Comp>.md`.
- Couleurs : n'utiliser que les variables CSS de `design/tagg/tokens.css` (`--forest`, `--terra`, `--sun`, `--moss`, `--danger`, `--bg-*`, `--line*`, `--ink*`, `--sport-*`, `--chart-*`). Aucun hex en dur dans le CSS, le TSX ou le Python ; côté Python passer par `theme.py` (= `design/tagg/theme_tokens.py`).
- Rôles : `forest` structure et actions, `terra` = « vous », `sun` = signal ponctuel, `moss` = positif en remplissage, `danger` = seul rouge. `moss` et `sun` ne sont jamais des couleurs de texte ; utiliser `*-ink` sur un fond `*-tint`.
- Typo : Manrope (texte) + DM Mono (chiffres, axes, surtitres, pastilles). Tout nombre est en mono, aligné à droite dans un tableau, unité en `--ink-muted` plus petite.
- Composants : classes `tm-*` de `design/tagg/components/components.css`. Ne pas créer de nouveau style de bouton, pastille ou carte sans l'ajouter d'abord au design system.
- Interdits : emoji dans l'UI, dégradés, ombres autres que `--shadow-card` / `--shadow-pop` / `--shadow-primary`, bord gauche coloré (sauf `.tm-session`), couleur de section (l'ancienne échelle `--scale-N` est supprimée).
- Graphiques Plotly : suivre `design/tagg/charts.md` (fond de carte, grille horizontale seule, séries `CHART_YOU_1..5`, références `CHART_REF` pointillé). Le titre d'un graphique vit dans la carte HTML, pas dans la figure.
- Nom du produit : **TAGG** (motto : Train · Analyse · Guide · Grow). Plus aucune occurrence de « TrailMetrics » dans l'UI.