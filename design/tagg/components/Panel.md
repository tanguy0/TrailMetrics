# Panel

Le panneau remplace `.panel`. Il porte un **numéro** (`index`) en mono forest à la place de l'ancienne couleur de section, le titre, les pastilles de source de données (`meta`), une description et ses actions ; son corps est une grille 12 colonnes de `PlotCard`.

- `meta` : d'abord la source (forest), puis les méta-données neutres. C'est là que vit l'information « d'où viennent les données » : jamais dans le titre.
- `actions` : l'état de précalcul en pastille sun, « Ajouter un graphique » en secondaire `sm`, puis un bouton d'options icône seule (⋯).
- Un panneau ne contient pas d'autre panneau ; deux panneaux sont séparés de `space-6`.
