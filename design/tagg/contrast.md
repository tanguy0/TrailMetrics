# Contraste et relief (v1.1)

## Le diagnostic

La v1.0 a bien installé les trois niveaux de surface, mais la page d'accueil les empile *trois fois à la même teinte* : page crème → carte blanche → huit tuiles blanches bordées. Résultat : tout est au même poids, l'œil n'a pas de point d'entrée, et les accents (terra, sun, moss) n'existent plus que dans des deltas de 11 px. Le vert et les gris tiennent la page ; le reste de la palette a disparu. Les autres écrans ont le même défaut en moins fort, parce qu'un graphique y apporte déjà de la couleur.

La correction n'est pas « plus de couleur partout » — ce serait revenir à l'échelle gold→rouge par section. C'est **une hiérarchie en quatre gestes**, du plus large au plus fin.

## 1. Une vedette par page

Chaque écran a **un** bloc qui prend `bg-hero` (le vert foncé du rail) : texte crème, chiffre clé en `sun`. C'est l'ancre visuelle ; tout le reste de la page se lit par rapport à lui.

| page | vedette |
| --- | --- |
| Accueil | l'en-tête athlète (`tm-hero`) : avatar, nom, et trois chiffres de la semaine en cours (km, D+, forme) dont un en sun |
| Entraînement | le bilan de la semaine courante dans la colonne de droite (`tm-hero` compact) ; les autres semaines restent crème |
| Analyses | aucune : les courbes sont déjà la vedette. L'`AccessGrid` de la landing visiteur a sa colonne « Avec Strava » en `tm-panel--tint` |
| Plan de course | le temps final estimé |

Jamais deux vedettes. Si tout est important, rien ne l'est.

## 2. Les tuiles deviennent plates dans une carte

À l'intérieur d'une carte blanche, une tuile KPI est **crème, sans bord, sans ombre** (`tm-kpi--flat`, fond `bg-tile`). La carte est blanche, les tuiles sont crème : les deux niveaux se distinguent à nouveau sans ajouter une couleur. Seule une tuile posée directement sur la page garde son bord et son ombre.

Par ailleurs `bg-page` passe de #f4f1ea à **#eee9de** : un cran plus profond, les cartes blanches se détachent sur toutes les pages à la fois.

## 3. Le chiffre qui compte prend sa couleur

Dans chaque carte, **une tuile — deux au maximum — est la tête d'affiche** et son chiffre prend une couleur de rôle (`tm-kpi--forest`, `--terra`, `--sun`, `--moss`). Les autres restent en `ink`. Les rôles ne changent pas :

- `forest` : volume, totaux, ce qui structure (distance totale, D+ cumulé)
- `terra` : « vous » et votre coût — allure, coût du dénivelé, RPE élevé, écart défavorable
- `sun-ink` : signal du moment — forme (CTL), record récent, prochaine course
- `moss-ink` : positif — ressenti, fraîcheur, RPE bas

Une tuile tête d'affiche peut porter une **sparkline** (`tm-kpi__spark`, 28 px, trait 1,6 px dans la couleur du chiffre) : c'est la façon la plus économe de remettre de la couleur et du mouvement dans une carte de chiffres.

## 4. Les sections ont une identité, pas une couleur

Chaque carte porte un **surtitre mono** (`tm-section__kicker`) et une icône au trait, tous deux dans la couleur de rôle de la section (`tm-section--terra`, `--sun`, `--moss` ; forest par défaut). C'est la seule trace de couleur dans un en-tête : pas de fond, pas de bord. Sur l'accueil : *Historique* forest, *Santé* terra (c'est vous), *Forme* sun, *Ressenti* moss, *Zones* terra.

## Au niveau du texte

- `tm-hl` : surlignage `sun-tint` du **seul** groupe de mots clé d'un paragraphe (« votre coût du dénivelé a baissé de 4 % »). Un par paragraphe, jamais une phrase entière.
- `tm-num--*` : un nombre en mono coloré dans une phrase (« 3 540 m de D+ »). Même règle : un ou deux par bloc.
- `tm-callout` : une note qui doit être lue (sun), une alerte (terra), un conseil (forest). Une par carte au plus ; remplace les `.note` grises.
- Les en-têtes de tableau, les dates, les unités restent muets : ce qui ne change pas ne prend pas de couleur.

## Ce qui ne change pas

Un seul bouton primaire par écran. Pas de bord gauche coloré hors `tm-session`. Pas de dégradé. `moss` et `sun` jamais en texte (toujours `-ink`). Et la règle de la vedette tient aussi pour la couleur : si une carte a déjà une tuile terra, son surtitre n'est pas terra aussi — une couleur par carte, une vedette par page.
