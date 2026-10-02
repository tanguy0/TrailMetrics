# Highlights

Les quatre outils de mise en relief à l'intérieur d'une carte, montrés ensemble parce qu'ils s'utilisent ensemble — et avec parcimonie.

- **`tm-section`** + `tm-section__kicker` : titre de carte avec un surtitre mono et une icône dans la couleur de rôle (`--terra`, `--sun`, `--moss`, forest par défaut). C'est toute l'identité colorée d'une section : pas de fond, pas de bord.
- **`tm-kpi--flat`** : dans une carte blanche, la tuile est crème, sans bord ni ombre. **`tm-kpi--forest/--terra/--sun/--moss`** colore le chiffre de la tuile tête d'affiche (une, deux au plus par carte). **`tm-kpi__spark`** : sparkline 28 px dans la couleur du chiffre.
- **`tm-num--*`** et **`tm-hl`** : un nombre coloré ou un groupe de mots surligné dans une phrase. Un ou deux par bloc. Le surlignage est toujours `sun-tint`, le texte reste `ink`.
- **`tm-callout`** (`--terra`, `--forest`) : la note à lire, au plus une par carte. Remplace les `.note` grises.

Règle de cohérence : une carte a une couleur dominante (sa section et sa tuile tête d'affiche partagent le rôle) ; le surlignage sun est transversal et ne compte pas.
