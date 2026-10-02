# Accès, paliers et architecture de l'information (v2)

## Les quatre paliers

TAGG a désormais quatre niveaux d'accès, chacun nommé dans le vocabulaire freemium du système (gratuit à chaque étape, sauf le dernier qui est un service) :

| palier | comment on y arrive | ce que ça ouvre | libellé |
| --- | --- | --- | --- |
| **Visiteur** | rien | Landing, Blog, Outils → *Planification de course* et *Évaluation du niveau* (utilisables, résultats non sauvegardés) | « Gratuit · maintenant » |
| **Compte** | e-mail + mot de passe | Accueil (dégradé), sauvegarde des plans et des zones, Outils complets en aperçu, Coaching en aperçu + formulaire | « Gratuit · avec un compte » |
| **Strava** | connexion Strava depuis le compte | Accueil complet, Outils → *Profil de pente* et *Durabilité*, Analyses | « Gratuit · avec Strava » |
| **Coaché** | demande acceptée par le coach | Coaching (carnet prévu/réalisé) | « Coaché par TAGG » |

Un palier contient les précédents. La connexion Strava ne crée plus d'identité : elle **s'attache** à un compte.

## Arrivée sur le site

- Session valide (cookie 30 jours) → `/home` directement. Pas de landing.
- Pas de session → landing : lockup + motto, une phrase, l'`AccessGrid` (colonne *Sans compte* : Planification de course, Évaluation du niveau, Blog ; colonne *Avec un compte* : Accueil, Outils complets, Coaching), puis deux boutons côte à côte : **Créer un compte** (primaire) et **Se connecter** (secondaire). Le bouton Strava disparaît de la landing : il vit dans l'Accueil du compte.

## Navigation (rail), nouvel ordre

1. **Accueil** — compte requis
2. **Outils** — ouvert, avec sous-onglets : *Planification de course* · *Évaluation du niveau* · *Profil de pente* (Strava) · *Durabilité* (Strava)
3. **Analyses** — Strava requis ; ne contient plus que les pages sur mesure (et le bouton « Nouvelle analyse »)
4. **Coaching** — coaché requis (aperçu + formulaire sinon) ; remplace *Entraînement*
5. **Blog** — ouvert

Renommages : *Plan de course* → **Planification de course** / *Race Planning* ; *Entraînement* → **Coaching** ; nouveau **Évaluation du niveau** / *Level Assessment*. Les clés de traduction changent de nom (`nav.tools`, `nav.coaching`, `tools.race_planning`, `tools.level`, `tools.gap_profile`, `tools.durability`).

Dans le rail, les items que le palier n'ouvre pas restent cliquables avec le cadenas (règle v1.1) et mènent à leur `Teaser`. Le groupe du rail dit le palier manquant : « Avec un compte », « Avec Strava », « Coaché par TAGG ».

## Outils : les analyses communes deviennent des pages dessinées

Le diagnostic est le bon : les deux analyses qui accrochent (courbe GAP, durabilité) étaient enterrées dans *Analyses* au milieu de panneaux génériques. Elles deviennent des **pages d'outil**, construites comme l'Accueil (hero compact, tuiles têtes d'affiche, un graphique principal, une phrase de synthèse en `tm-hl`) et non comme un empilement de panneaux :

- **Profil de pente** (`/tools/gap`) : hero « votre coût du dénivelé », tuiles coût montée (terra) / coût descente / allure plat, la courbe GAP (famille *fonction*), une phrase « Vous perdez 4,8 % de moins que la référence en montée ». Le panneau *Courbes GAP* reste disponible dans Analyses pour ceux qui veulent le paramétrer.
- **Durabilité** (`/tools/durability`) : hero « votre dérive sur l'effort long », tuiles dérive à 2 h / à 4 h / confiance, les deux graphiques, phrase de synthèse.
- **Planification de course** (`/tools/race-planning`) : la page actuelle, inchangée, sauvegarde conditionnée au compte.
- **Évaluation du niveau** (`/tools/level`) : nouvelle, voir `level.md`.

Les trois pages « par défaut » d'Analyses (Simulateur GAP, Comparateur, Progression) ne sont plus créées automatiquement pour un nouvel athlète : elles deviennent des **modèles** proposés par « Nouvelle analyse → à partir d'un modèle ». Analyses redevient ce que son nom dit : des analyses faites sur mesure.

## Accueil dégradé (compte sans Strava)

La page garde exactement sa structure. Le hero affiche nom et e-mail, un kicker « Compte TAGG », et ses trois stats en **tiret** (`—`) ; à la place de l'action « Importer », le bouton primaire **Connecter Strava**. Les cartes gardent leur titre, leur surtitre et leur rôle, mais leurs tuiles sont des tirets et leurs graphiques des structures vides (la grille `chart-grid` sans série) ; chaque carte porte une ligne `body-sm` muted « Se remplit dès que Strava est connecté ». La carte *Zones* est l'exception : elle est **pleine** si l'athlète a fait une évaluation du niveau, et renvoie vers l'outil sinon (« Estimez vos zones sans Strava → »). Le `Teaser` n'est pas utilisé ici : on montre la vraie page, vide, avec une seule invitation en haut — pas une carte flottante par-dessus.

## Coaching (aperçu)

C'est la page qui doit donner envie : elle est **stylée comme une offre**, pas comme une page verrouillée. Hero `bg-hero` avec un titre fort (« Un plan construit sur vos données, pas sur un modèle »), trois points en `tm-teaser__list` (plan hebdo adapté à votre historique Strava, échanges avec le coach dans le carnet, ajustements chaque semaine), un aperçu réel du carnet en arrière-plan (structure vide), puis le **formulaire de demande** dans une carte : message libre, et un moyen de contact — e-mail pré-rempli depuis le compte, téléphone facultatif, l'un des deux requis. Bouton primaire « Envoyer ma demande ». Après envoi : la carte devient un état « Demande envoyée le 2 oct. — le coach vous répond par e-mail ou téléphone », avec possibilité de modifier le message tant qu'elle est en attente. Pour un coaché : la page Coaching actuelle (carnet).

## Coach (toi)

Dans Coaching, un compte `coach` voit en plus une carte **Athlètes** avec deux listes : *En attente* (nom, e-mail, téléphone, message, date ; boutons Accepter / Décliner) et *Coachés* (nom, dernière activité, lien « Voir comme »). Accepter crée le lien coach→athlète et débloque la page pour lui ; le sélecteur « Athlète » du rail s'alimente de cette liste au lieu de la variable d'environnement. Décliner garde la demande dans un état fermé (pas de suppression).
