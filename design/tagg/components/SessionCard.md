# SessionCard

Les trois cellules du calendrier d'entraînement, dans une colonne `tm-day` (aujourd'hui = fond `forest-tint`, numéro en pilule forest).

- `kind: "done"` (défaut) : séance Strava. Bord gauche 3 px coloré par `sport` (`run` forest, `bike` lac, `hike` ocre, `swim` prune) — la seule exception à la règle « pas de bord gauche », justifiée parce que le titre et l'icône redisent le sport. Stats en mono (distance en `ink`, temps et D+ en muted), pastilles RPE (moss ≤ 4, sun 5–7, danger ≥ 8) et ressenti.
- `kind: "planned"` : séance ou note prévue, bord pointillé `line-strong`, fond `bg-surface-alt`, corps en `body-sm`.
- `kind: "goal"` : objectif, fond `terra-tint` + texte `terra-ink` ; `secondary` passe en sun.
- Le consommateur fournit l'icône de sport (SVG 14 px au trait) et les chaînes déjà formatées.
