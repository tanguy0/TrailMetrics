# NavRail

Le rail remplace `.sidebar`. Fond `bg-rail` (forest foncé), `radius-xl`, posé à `space-8` du bord de fenêtre avec `shadow-card`. C'est le seul aplat de couleur de marque de l'écran.

- `brand` : le lockup `tagg-lockup-on-rail.svg` à 28 px de haut. Jamais le motto dans le rail.
- `items` : icône 17 px au trait + libellé 13,5 px. Actif = fond blanc 12 % + gras. Visiteur non connecté : deux groupes `tm-rail__group` (« Ouvert », « Avec Strava ») ; les items du second restent des liens, vers leur `Teaser`, avec un cadenas 14 px (`tm-rail__link--locked`) — plus d’opacité 45 % (voir `visitor.md` § Rail).
- `switcher` : le sélecteur coach / athlète, en bouton plein largeur sous la marque.
- `count` (`tm-rail__count`) : pastille de notification au bout d'un lien du rail — un nombre en mono 11 px sur fond `sun` (signal ponctuel), texte `ink`. Pour un coach : les demandes de coaching en attente, sous le sélecteur. Absente à zéro, jamais un « 0 ».
- `footer` : un indicateur de charge ou l’état de synchro Strava, poussé en bas ; pour un visiteur, le bouton Strava `sm` pleine largeur.
- Largeur fixe 236 px ; sur mobile le rail devient une barre basse de 5 icônes (non prévisualisée ici).
