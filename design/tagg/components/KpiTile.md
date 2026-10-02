# KpiTile

La tuile KPI remplace `.tile` et `.metric`. Libellé en `caption` muted, valeur en `num-xl` mono, unité en `body-sm` muted sur la même ligne de base, delta en mono 11,5 px.

- `trend` colore uniquement le delta : `up` forest, `down` terra, `signal` sun-ink. La valeur reste toujours en `ink`.
- Les tuiles se posent en rangée de 3 ou 4 sur `bg-page` ; à l'intérieur d'un panneau on utilise une `PlotCard` sans ombre.
- Le consommateur formate la valeur (séparateur de milliers insécable, `×`, `:`) : la tuile n'arrondit rien.

- La valeur prend `tm-kpi__num--lg` au-delà de 5 caractères et `--sm` au-delà de 9 (le composant React le fait seul) ; valeur et unité ne passent jamais à la ligne. Voir *Densité des cellules*.
