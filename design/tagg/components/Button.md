# Button

Le bouton a trois niveaux (primaire, secondaire, fantôme) plus un danger et un Strava. Un écran n'a qu'un primaire visible à la fois ; les actions de panneau sont en secondaire taille `sm`.

- `variant` : `primary` (forest, `shadow-primary`), `secondary` (blanc bordé `line-strong`), `ghost` (texte forest, fond `forest-tint` au survol), `danger` (bordé danger, jamais plein), `strava` (couleur imposée, écran de connexion uniquement).
- `size` : `md` 40 px ou `sm` 32 px. `wide` étire à 100 % (formulaires).
- `icon` : un SVG 16 px au trait, à gauche du libellé ; l'icône seule exige `aria-label`.
- Le consommateur fournit le libellé en infinitif (« Lancer », « Exporter ») et l'icône.
