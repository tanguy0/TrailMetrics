# Hero

La **vedette** d'une page : le seul bloc en `bg-hero` (le vert foncé du rail). Il ancre l'écran ; tout le reste se lit par rapport à lui. Une page n'en a qu'un, certaines n'en ont pas (Analyses).

- `kicker` : le contexte temporel en mono (semaine, fenêtre). `title` : qui ou quoi. `meta` : l'état (synchro, nombre d'activités).
- `stats` : deux ou trois chiffres au plus, en mono 26 px ; **un seul** porte `key: true` et passe en `sun` — c'est le chiffre de la semaine. Les autres restent crème.
- `action` : un bouton secondaire au plus (la version sur fond sombre est gérée par le composant). Jamais le bouton primaire ici : le primaire de la page vit dans une carte.
- **`tm-hero--compact`** : la version colonne pour une cellule étroite (bilan de la semaine courante sur Entraînement, temps visé sur Plan de course). Elle garde **toutes** les lignes du bilan qu'elle remplace — on ne fait pas disparaître une donnée pour faire une vedette — et un seul chiffre passe en `is-key`. Un `tm-hero__sep` sépare les groupes de lignes.

Sur fond `bg-hero`, les seuls blancs translucides autorisés sont `on-hero-line` (bords) et `on-hero-fill` (fonds). Pas de décor : le hero est un bloc plein, sans forme ni dégradé.

## Contenu du hero de Plan de course

Le titre du hero n'est jamais le libellé d'une stat (« Temps visé » n'apparaît qu'une fois, dans la stat `is-key`). Structure : `kicker` = « Plan de course · courbe {curve_label} » (+ « · personnalisée » quand elle l'est) ; `title` = le nom de la course ; `meta` = la stratégie en une ligne, composée des champs du résumé : « GAP {gap_pace} /km · réel {average_pace} /km · {section_count} sections · {aid_station_count} ravitos · dérive {durability_multiplier_finish} à l'arrivée ({confidence}) » — chaque fragment absent du résumé est simplement omis. Stats : Temps visé (`is-key`), Distance, D+ / D− (`+2 100 / −1 980`), Allure GAP. Quatre stats au maximum.
