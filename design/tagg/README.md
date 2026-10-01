# TAGG

**Train · Analyse · Guide · Grow.** TAGG (ex-TrailMetrics) est un atelier d'analyse de données de course à pied et de trail : l'athlète construit ses analyses à partir de ses propres activités, et l'app les met en forme. Ce système décrit comment TAGG doit se présenter : une **plateforme produit** calme et lisible, où les chiffres et les courbes sont les vedettes et où l'interface s'efface.

## Principes

1. **Les données sont le décor.** Un écran TAGG se reconnaît à ses courbes et à ses chiffres en mono, pas à ses boutons. Tout ce qui n'est pas une donnée est neutre : blanc, crème grisé, filets fins.
2. **Un seul accent chaud à la fois.** `forest` structure (marque, actions, onglets). `terra` veut dire « vous » et ne sert qu'à ça. `sun` est un signal ponctuel (record, infobulle, précalcul). On ne pose jamais terra et sun côte à côte sur la même carte sans raison.
3. **La hiérarchie vient des surfaces, pas des couleurs.** Fond `bg-page` → panneau blanc avec `shadow-card` → plot-card bordée sans ombre. Trois niveaux, jamais quatre.
4. **Les chiffres sont alignés et tabulaires.** Toute valeur numérique est en `mono` (DM Mono), alignée à droite dans un tableau, avec son unité en `ink-muted` et plus petite.
5. **Pas d'emoji, pas de dégradé, pas de bord gauche coloré** (sauf la carte de séance, où le bord code le sport et porte toujours un libellé).

## Couleurs

Les coloris phare de TrailMetrics sont conservés à l'identique : `forest` #2e6f40, `terra` #c65d3b, `sun` #e8a33d, `moss` #5e9c4e, `danger` #8e2c18. Ce qui change : les neutres sont re-hiérarchisés (page grisée, cartes blanches), l'ancienne échelle gold→rouge par section (`--scale-1..6`) disparaît — un panneau est numéroté, pas coloré — et chaque accent gagne une variante `-tint` pour les fonds et `-ink` pour le texte qui doit passer 4,5:1.

| rôle | tokens | règle |
| --- | --- | --- |
| structure & marque | `forest`, `forest-hover`, `forest-tint` | bouton primaire, lien, onglet actif, aujourd'hui, série 1 |
| vous | `terra`, `terra-ink`, `terra-tint` | série 2, objectif principal, écart défavorable |
| signal | `sun`, `sun-ink`, `sun-tint` | record, infobulle, précalcul à jour, objectif secondaire |
| positif | `moss`, `moss-ink`, `moss-tint` | barres de volume, RPE ≤ 4, toggle actif |
| alerte | `danger`, `danger-tint` | suppression, erreur, RPE ≥ 8 |
| sports | `sport-run` `sport-bike` `sport-hike` `sport-swim` `sport-other` | bord de carte de séance + libellé texte toujours présent |

Contraste : `ink-faint` (#7d7366) fait 4,6:1 sur blanc mais 4,1:1 sur `bg-page` ; il ne sert donc qu'à l'intérieur des cartes. `terra` en texte pur fait 4,2:1 : réservé au ≥ 14 px semi-gras, sinon `terra-ink`. `moss` et `sun` ne sont jamais des couleurs de texte.

## Typographie

Deux familles hébergées Google Fonts : **Manrope** (400–800) pour tout le texte et **DM Mono** (400, 500) pour les chiffres, axes, surtitres et pastilles. Chargement :

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap">
```

Les titres Manrope sont serrés (`letter-spacing` −0.02 à −0.035em) ; le corps reste à 0. Les styles `kicker` et `label` sont les seuls en capitales.

## Espacements, rayons, ombres

Grille de 4 px. Les rayons montent avec le niveau de surface : `radius-sm` 6 pour ce qui est dans une carte, `radius-md` 10 pour les contrôles, `radius-lg` 12 pour une plot-card, `radius-xl` 16 pour un panneau ou le rail. Une seule ombre par niveau (`shadow-card`), une pour ce qui flotte (`shadow-pop`). Le focus clavier est toujours `focus-ring` + bordure `forest`.

## Iconographie

Icônes au trait 1,75 px, chapeaux et jointures ronds, grille 24, couleur `currentColor` — le style Lucide. Taille 16 px dans les boutons et pastilles, 17 px dans le rail, 14 px dans une carte de séance. Jamais d'emoji (le menu actuel 🏠📊📅🏁📰 est remplacé par home / chart / calendar / flag / newspaper).

## Courbes et graphiques

La règle complète est dans la section **Graphiques (Plotly)**. En bref : fond de la carte, grille horizontale seule en `chart-grid`, axes en `chart-axis` 11 px mono, séries de l'athlète en `chart-you-1..5`, références en `chart-ref` pointillé, point actif avec halo `forest-tint`, infobulle = carte `bg-surface` + `shadow-pop`.

## Composants

Les composants sont décrits et prévisualisés dans l'onglet Composants. Ils existent en deux formes équivalentes : des **classes CSS** préfixées `tm-` dans `components/components.css` (pour l'app actuelle en CSS écrit main) et des **composants React** `window.TAGG.*` dans `components/components.js` qui ne font que poser ces classes. Le consommateur fournit le texte, les icônes (SVG inline) et les données ; le système fournit la forme.

Deux composants servent le visiteur sans compte (`Teaser`, `AccessGrid`) ; la logique d'ensemble est dans la section **Expérience visiteur (freemium)**.

Correspondance avec le code existant : voir la section **Migration depuis globals.css**.
