# Expérience visiteur (freemium)

Un visiteur sans compte doit comprendre en une page **ce qu'il peut faire tout de suite** et **ce qu'il aurait en connectant Strava**, et il doit pouvoir cliquer sur les deux. Le modèle mental est celui d'un freemium à deux niveaux — « Gratuit · maintenant » et « Gratuit · avec Strava » — alors que tout est gratuit : on emprunte la clarté du freemium (montrer le niveau supérieur, le rendre désirable, dire exactement comment y accéder), pas son vocabulaire commercial. Jamais « Pro », « Premium », « Débloquer », « Offre ».

## Les trois surfaces

1. **Landing (`/`)** — remplace le texte actuel « data-science workbench ». Dans l'ordre : lockup + motto, une phrase (« TAGG analyse vos sorties et vous aide à progresser. Deux outils sont ouverts à tous ; le reste s'ouvre en connectant Strava. »), puis l'`AccessGrid`, puis le bouton Strava et sa phrase de confiance. Le visiteur voit d'abord ce qui est à lui.
2. **Rail** — tous les items restent visibles et cliquables. Deux groupes séparés par un libellé `tm-rail__group` : *Ouvert* (Plan de course, Blog) puis *Avec Strava* (Accueil, Analyses, Entraînement) ; les seconds portent un cadenas 14 px à droite (`tm-rail__link--locked`) et ne sont plus en opacité 45 %. En bas du rail, à la place de l'indicateur de charge : le bouton Strava en `sm` plein largeur.
3. **Pages verrouillées** (`/home`, `/pages`, `/training`) — plus de `redirect("/")`. La page rend son `Teaser` : la vraie page en arrière-plan avec des données d'exemple si disponibles (sinon sa structure vide), la carte par-dessus avec trois bénéfices et le bouton Strava. Le visiteur garde l'URL, le rail, le contexte.

## Données d'exemple (recommandé)

L'architecture « une analyse est une donnée » rend un **athlète de démonstration** peu coûteux : un jeu d'activités anonymisées précalculées, servi en lecture seule. Avec lui, « Voir un exemple » ouvre la vraie page Analyses ou Entraînement, fonctionnelle, avec une pastille `sun` « Données d'exemple » dans le `PageHeader` et le bouton Strava en action primaire. C'est la version la plus convaincante du teaser : le visiteur manipule le produit. Sans jeu de données, le `Teaser` seul suffit pour la PR de refonte ; l'athlète de démonstration peut venir ensuite.

## Pages ouvertes

Plan de course et Blog ne changent pas de comportement, mais signalent le niveau supérieur sans l'imposer : une seule ligne discrète en fin de page (`body-sm`, `ink-muted`) — « Avec Strava, TAGG apprend ce plan à partir de vos propres montées » — avec un lien texte, pas un bandeau ni une modale. Zéro interruption sur ce qui est ouvert.

## Mesure du succès

Le visiteur a réussi s'il a fait quelque chose (simulé un plan, lu un article) *avant* de décider de connecter Strava, et s'il sait, sans l'avoir fait, ce que la connexion lui donne. Le taux de connexion Strava au premier passage n'est pas l'objectif : la compréhension l'est.
