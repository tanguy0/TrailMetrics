# Évaluation du niveau — estimation des zones (v2)

Une page, trois façons d'entrer, un seul pipeline caché : **tout converge vers une vitesse à VO₂max (VMA, km/h)**, puis les zones de l'Accueil s'en déduisent avec les pourcentages déjà en place (`VMA_PACE_ZONES` : 60–65, 70–75, 85–90, 95–100, 105–115 % ; FC par `HR_ZONE_MAX_PCT` si FCmax connue). Le calcul est côté API (`src/domain/level/`) pour que l'Accueil, l'outil et plus tard le coaching lisent la même valeur.

## Pivot commun : VDOT (Daniels & Gilbert)

Pour une performance de durée `t` (minutes) à vitesse `v` (m/min) :

```
VO2(v)      = -4.60 + 0.182258·v + 0.000104·v²
pct(t)      = 0.8 + 0.1894393·e^(-0.012778·t) + 0.2989558·e^(-0.1932605·t)
VDOT        = VO2(v) / pct(t)
```

La **VMA** est la vitesse `v*` telle que `VO2(v*) = VDOT` (racine positive du second degré), en km/h = `v*·60/1000`. Vérifier par des tests contre des valeurs connues : 5 km en 20:00 → VDOT ≈ 49,8 ; 10 km en 40:00 → ≈ 52,0 ; marathon en 3:00:00 → ≈ 53,5 ; 1500 m en 5:00 → ≈ 50,4.

## Les trois entrées

**1. Demi-Cooper (6 min).** Entrée : distance (m). `t = 6`, `v = d/6`. Deux estimations sont possibles : la convention de terrain VMA = `d/100` km/h, et la voie VDOT. L'écart est faible (< 3 %) ; on affiche la voie VDOT pour rester cohérent avec les autres entrées, et la convention de terrain en note (« test de terrain : 16,2 km/h »).

**2. Vitesse critique (3 min + 12 min).** Entrées : `d3` et `d12` (m).
```
CS  = (d12 − d3) / (720 − 180)       m/s   vitesse critique
D'  = d3 − CS·180                    m     réserve anaérobie
```
Validation : `D'` entre 50 et 500 m et `CS` entre 1,5 et 7 m/s, sinon message « les deux distances ne sont pas cohérentes ». Puis on passe par le pivot : la performance sur 12 min est injectée dans VDOT (`t = 12`, `v = d12/12`) — c'est la meilleure estimation de la VMA à partir de ce test ; on affiche aussi CS (en allure, « seuil critique ») et D' comme informations propres au test, parce qu'elles valent pour un coureur et n'existent pas ailleurs dans l'app. (La relation CS ≈ 0,9 × VMA est donnée en note de cohérence, pas utilisée pour calculer.)

**3. Records.** Entrées : une ou plusieurs lignes `distance` (m, libre, avec suggestions 1 000 / 1 609 / 3 000 / 5 000 / 10 000 / 21 097 / 42 195) + `temps`. Validation : allure entre 2:00 et 12:00 /km, durée ≥ 3 min (en dessous le modèle ne vaut rien), ≤ 6 h. Chaque record donne un VDOT ; la VMA retenue est celle du **VDOT médian** quand il y a ≥ 3 records, de la **moyenne** à 2, du seul à 1. On affiche la dispersion : « vos records sont cohérents (±1,2 VDOT) » ou « votre 5 km est nettement meilleur que votre marathon : la VMA retenue vient du médian ; vos zones d'endurance sont peut-être optimistes ».

## Sortie

Une structure unique `LevelEstimate { method, vma_kmh, vma_pace_s_per_km, vdot, confidence, notes[], zones[] }` ; sauvegardée dans `level_estimates(account_id, method, inputs jsonb, result jsonb, created_at)` quand un compte existe ; la dernière estimation devient `vma_pace_s_per_km` de la carte *Zones* de l'Accueil (qui gagne un libellé « estimée le 2 oct. · méthode : records »). Sans compte : résultat affiché, un `tm-callout` forest « Créez un compte pour garder vos zones ».

## Page `/tools/level`

Hero compact : « Où en êtes-vous ? » ; trois onglets segmentés *Demi-Cooper* · *Vitesse critique* · *Records* ; formulaire court (`Field`, `Button` primaire « Estimer ») ; résultat sous forme de carte : tuile VMA tête d'affiche (forest, `tm-kpi__num--lg`), tuile allure VMA, tuile VDOT (muted), puis la grille des cinq zones d'allure (identique à la carte Zones de l'Accueil) et la carte FC si FCmax saisie. Une phrase de synthèse avec un `tm-hl`. Les notes de cohérence en `tm-callout`.
