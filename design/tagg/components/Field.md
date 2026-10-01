# Field

Champ texte et sélecteur avec leur libellé, pour `ParamForm`, `DataSourceEditor` et `PanelEditor`. Libellé en `caption` muted au-dessus, contrôle 40 px bordé `line-strong`, focus = bordure forest + `focus-ring`.

- Passer `options` rend un `<select>` (chevron intégré en CSS) ; sinon un `<input>`.
- Le `Toggle` (composant séparé, même carte) remplace les cases à cocher binaires du formulaire ; les listes à choix multiples gardent des cases natives avec `accent-color: var(--forest)`.
- Les sous-paramètres conditionnels s'indentent de `space-4` et prennent un filet gauche `line`, pas de fond.
