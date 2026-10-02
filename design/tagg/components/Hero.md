# Hero

La **vedette** d'une page : le seul bloc en `bg-hero` (le vert foncé du rail). Il ancre l'écran ; tout le reste se lit par rapport à lui. Une page n'en a qu'un, certaines n'en ont pas (Analyses).

- `kicker` : le contexte temporel en mono (semaine, fenêtre). `title` : qui ou quoi. `meta` : l'état (synchro, nombre d'activités).
- `stats` : deux ou trois chiffres au plus, en mono 26 px ; **un seul** porte `key: true` et passe en `sun` — c'est le chiffre de la semaine. Les autres restent crème.
- `action` : un bouton secondaire au plus (la version sur fond sombre est gérée par le composant). Jamais le bouton primaire ici : le primaire de la page vit dans une carte.
- Sur Entraînement, la version compacte (sans avatar, stats en colonne) remplace le bilan crème de la semaine courante uniquement.
- **Pas de décor, bloc plein.** Aucun motif, rond ni dégradé : un fond `bg-hero` uni. Sur ce fond, les bords prennent `on-hero-line` (cadre de l'avatar, bouton secondaire) et les fonds `on-hero-fill` ; aucune autre transparence en dur.
