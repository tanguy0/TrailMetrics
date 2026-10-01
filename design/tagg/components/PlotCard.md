# PlotCard

La carte de graphique vit **dans** un panneau : bord `line`, `radius-lg`, pas d'ombre, fond blanc. Titre `title-sm`, sous-titre `body-sm` muted (unité, fenêtre, lissage), pastille neutre avec la catégorie du plot registry (Évolutions, Records, Modèles…).

- `span` : 4 à 12 colonnes sur 12. Combinaisons usuelles : 8 + 4, 7 + 5, 6 + 6, 5 + 4 + 3.
- La figure Plotly prend `bg-chart` et aucune marge haute : le titre est celui de la carte (voir *Graphiques*).
- `legend` : une entrée par série, trait de 18 px ; la classe `is-ref` rend le pointillé des références.
- Les actions de carte (paramètres, dupliquer, supprimer) apparaissent au survol dans l'en-tête, en boutons icône `sm`.
