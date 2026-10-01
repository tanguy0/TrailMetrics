# Teaser

La page « verrouillée » d'un visiteur sans compte : au lieu d'une redirection ou d'un lien inerte, on affiche la vraie page en arrière-plan (désaturée, dégradée vers le bas, non cliquable) et, par-dessus, une carte qui dit ce qu'on y trouverait et comment y accéder.

- `background` : la page elle-même rendue avec des **données d'exemple** (athlète de démonstration) si le dataset existe, sinon la structure vide (tuiles, grille de graphique). Jamais un screenshot flou : l'arrière-plan doit avoir le même DOM que la vraie page pour rester synchrone avec le produit.
- `kicker` : toujours le niveau d'accès, dans le vocabulaire freemium : « Gratuit · avec Strava ». Jamais « Pro », « Premium » ou « payant » — l'app est gratuite.
- `bullets` : trois bénéfices concrets maximum, verbes à l'infinitif ou noms, pas de marketing.
- `actions` : le bouton Strava en primaire et, si des données d'exemple existent, « Voir un exemple » en fantôme qui ouvre la même page en mode démo (pastille « Données d'exemple » en haut).
- `fine` : la phrase de confiance (gratuit, sans carte, ce qui est lu).
