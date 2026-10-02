# Densité des cellules et des valeurs

Une valeur qui passe à la ligne dans une tuile ou une cellule casse la grille et la lecture. Ces règles s'appliquent à **toute** valeur formatée (tuile KPI, cellule de tableau, bilan de semaine, stats de hero).

## Dates

- Jamais d'ISO (`2026-09-30`) dans l'interface : c'est un format machine. `formatDate` devient contextuel :
  - **court** (tuiles, cellules, pastilles) : `30 sept.` ; l'année n'apparaît que si elle diffère de l'année courante, en deux chiffres : `30 sept. 25`.
  - **relatif** (hero, meta, états de synchro) : `il y a 12 min`, `hier`, `lundi`, puis le format court au-delà de 7 jours.
  - **long** (titre de page, PDF) : `mardi 30 septembre 2026`.
- Un intervalle de dates : `29 sept. – 5 oct.` (tiret demi-cadratin, espaces fines autour), `nowrap`.
- En mono 12 px muted dans les cellules (`is-date`), jamais en gras.

## Nombres, allures, durées

- Séparateur de milliers = espace fine insécable (`3 540`), unité séparée par une espace insécable, `tabular-nums` partout en mono.
- **Intervalles d'allure** : `6:17–6:48`, sans espaces autour du tiret, le plus rapide en premier, `nowrap`. L'unité (`/km`) quitte la valeur et monte dans le libellé : « Allure (/km) ». Si la cellule reste trop étroite, on **ne coupe pas** : la colonne s'élargit (`min-width: 9ch`) ou le tableau défile horizontalement dans sa carte.
- Durées : `1:12` sous l'heure, `4:12:08` au-dessus, `1 062 h` au-delà de la journée ; jamais `01:12:00`.
- Une valeur et son unité ne se séparent jamais : `white-space: nowrap` sur `.tm-kpi__value`, `.is-num`, `.tm-hero__stat .v`.

## La valeur de tuile s'adapte à sa longueur

Le chiffre d'une tuile n'a pas une taille, il en a trois, choisies par le nombre de caractères de la valeur (unité exclue) :

| caractères | taille | exemples |
| --- | --- | --- |
| ≤ 5 | 28 px (`num-xl`) | `284`, `1.62×`, `4:21` |
| 6 – 9 | 22 px (`num-lg`) | `11 240`, `6:17–6:48`, `1:31:04` |
| > 9 | 17 px | `30 sept. 25`, un libellé |

En CSS : `tm-kpi__num--lg`, `tm-kpi__num--sm` posés par le composant selon `value.length`. Une tuile ne contient qu'une valeur ; une donnée qui demande deux lignes de valeur (« de… à… ») devient deux tuiles ou une tuile + delta.

## Texte dans une cellule

- Noms d'activité, titres : une ligne, `text-overflow: ellipsis`, le texte complet en `title`.
- Une cellule n'a pas de retour à la ligne volontaire : une donnée secondaire va dans le delta (tuile) ou dans une colonne (tableau), pas sous la valeur.
- Les libellés de tuile peuvent passer sur deux lignes ; les valeurs jamais.
