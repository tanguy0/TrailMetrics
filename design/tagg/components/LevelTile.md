# LevelTile

Le niveau d'un coureur face à une référence, lisible d'un coup d'œil : une tuile par terrain ou par qualité (profils GAP et durabilité), sur l'échelle commune à cinq niveaux. Classes `tm-level` dans une grille `tm-level-grid`.

- Contenu : une icône (SVG inline, style Lucide, 22 px), un libellé court en mots de coureur (« Montée raide », « Efforts longs » — jamais de seuil ni de plage chiffrée), le mot du niveau en grand, puis une jauge de cinq segments (`tm-level__meter`, segments `is-on` de gauche à droite : 1 = faible … 5 = excellent). Pas de phrase d'explication dans la tuile : la jauge et la couleur suffisent.
- Le niveau teinte toute la tuile (`tm-level--<niveau>`) : `excellent` / `good` → moss, `average` → forest, `limited` → sun, `poor` → danger, `insufficient` → fond de tuile neutre, jauge vide, mot en `ink-muted`. Fond en `*-tint`, mot et icône en `*-ink`, segments pleins dans l'accent lui-même : `moss` et `sun` restent des remplissages, jamais du texte.
- Pas de bord coloré ni d'ombre : la tuile vit à l'intérieur d'une carte.
