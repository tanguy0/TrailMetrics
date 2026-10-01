# DataTable

Tableau de données : en-têtes `label` capitales en `ink-faint`, lignes alternées `bg-surface-alt`, survol `forest-tint`, filets `line`.

- Colonnes `numeric` : alignées à droite, en `num` mono 13 px — toujours, sans exception, pour que les chiffres se comparent verticalement.
- Colonnes `date` : mono 12 px muted.
- `best` sur une ligne colore ses nombres en forest (remplace `.cell--best`).
- Le tableau est lui-même une carte (`radius-xl`, `shadow-card`) quand il est posé sur la page ; dans un panneau, le consommateur retire l'ombre avec `.tm-plot .tm-table{box-shadow:none}`.
