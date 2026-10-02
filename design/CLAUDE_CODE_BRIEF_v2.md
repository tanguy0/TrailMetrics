# Brief Claude Code — TAGG v2 : comptes, paliers d'accès, Outils, Coaching

Trois PR, **dans cet ordre**, chacune livrable seule. Specs dans `design/specs/` (`access.md`, `auth.md`, `level.md`, `coaching.md`) ; design dans `design/tagg/` (fusionner `README.md`, `access.md`, coller `components.v2.additions.css` en fin des deux `components.css`). Maquettes de référence sur le canvas : *Connexion*, *Accueil dégradé*, *Outils · Évaluation du niveau*, *Coaching · page-offre*.

## PR 1 — Comptes et sessions (`auth.md`)
1. Schéma : `accounts`, `sessions`, `password_resets`, `login_attempts`, `athletes.account_id` ; migration SQL idempotente dans `schema.sql`.
2. API : `auth/register`, `auth/login`, `auth/logout`, `auth/logout-all`, `auth/reset` (+ `MailSender` Resend/SMTP, repli « écrivez à MASTER_EMAIL »), argon2id, sessions opaques hachées, limitation de débit, messages d'erreur identiques, en-têtes de sécurité. `api/deps.py` : `current_account` → `current_athlete_id` via `athletes.account_id`. `role` remplace `COACH_ATHLETE_IDS`/`MASTER_EMAIL` (migration : l'e-mail master reçoit `master`).
3. OAuth Strava **depuis un compte** : les trois cas de rattachement d'`auth.md` ; déconnexion Strava.
4. Web : `/login`, `/register`, `/reset`, `/reset/[token]` (`tm-auth`, segment Se connecter / Créer un compte, shell réduit au lockup) ; middleware : session → `/` redirige vers `/home` ; landing v2 (deux boutons : Créer un compte primaire, Se connecter secondaire ; Strava retiré de la landing).
5. Accueil dégradé : hero avec tirets et bouton **Connecter Strava** en primaire, tuiles `tm-dash`, `tm-empty-chart`, ligne « Se remplit dès que Strava est connecté » par carte ; la carte Zones renvoie vers l'outil (`access.md`).
6. Tests listés dans `auth.md`. Le fichier `.env.example` documente `MAIL_*`.

## PR 2 — Outils (`access.md`, `level.md`)
1. Rail v2 : Accueil · **Outils** · Analyses · **Coaching** · Blog, groupes par palier (« Avec un compte », « Avec Strava », « Coaché par TAGG »), cadenas cliquables. Renommages et nouvelles clés de traduction (`nav.tools`, `nav.coaching`, `tools.*`), fr + en.
2. `/tools` avec sous-onglets : `/tools/race-planning` (page actuelle déplacée, sauvegarde conditionnée au compte), `/tools/level` (nouveau), `/tools/gap`, `/tools/durability` (pages dessinées — hero compact, tuiles têtes d'affiche, graphique principal, phrase `tm-hl` — réutilisant les plots `gap_curve` et `durability_curve` existants ; `Teaser` si pas de Strava).
3. `src/domain/level/` : VDOT (Daniels-Gilbert), les trois entrées, validations, `LevelEstimate` ; tests contre les valeurs connues de `level.md` ; table `level_estimates` ; la dernière estimation alimente `vma_pace_s_per_km` et la carte Zones (« estimée le … · méthode »). Les pourcentages de zones restent ceux de `HomeScreen` — les **déplacer** dans `src/domain/level/zones.py` et les servir par l'API pour qu'il n'y ait qu'une définition.
4. Analyses : les trois pages par défaut ne sont plus créées automatiquement ; elles deviennent des **modèles** dans « Nouvelle analyse → à partir d'un modèle ». Les athlètes existants gardent les leurs.
5. Visiteur : `/tools/race-planning` et `/tools/level` utilisables sans session ; résultat affiché + `tm-callout` « Créez un compte pour garder … ».

## PR 3 — Coaching (`coaching.md`)
1. Tables `coaching_requests`, `coaching` ; prédicat `is_coached` ; routes demande / retrait / accept / decline (rôle coach|master).
2. Page Coaching non coaché : `tm-offer` (hero-offre, trois points, chiffres de preuve, aperçu du carnet en structure vide) + formulaire (`message`, contact e-mail pré-rempli ou téléphone, l'un requis) ; états *à remplir* / *envoyée le …* (modifier, retirer) / *déclinée*.
3. Coach : carte **Athlètes** (En attente : Accepter / Décliner ; Coachés : dernière activité, Voir comme) ; le sélecteur « Athlète » du rail lit `coaching`. Notification e-mail au coach via `MailSender` si configuré.
4. Les chiffres de preuve de la page-offre (« 12 athlètes coachés… ») viennent de la base (`count(coaching)`), jamais écrits en dur ; si < 3, la ligne est masquée.

## Garde-fous
- Rien de Strava n'entre dans `accounts` ; rien du compte ne va dans les tables clés par `athlete_id` sauf `account_id`.
- Vocabulaire des paliers : « Gratuit · maintenant / avec un compte / avec Strava », « Coaché par TAGG ». Jamais Pro/Premium.
- Une vedette par page : l'Accueil dégradé garde son hero, la page Coaching a `tm-offer` et rien d'autre en `bg-hero`.
- Aucune couleur hors `tokens.json` ; `components.css` fusionné, jamais remplacé.

## Prompt (PR 1)
> Lis `design/specs/auth.md` et `design/specs/access.md`, puis `design/CLAUDE_CODE_BRIEF_v2.md`. Nous faisons la **PR 1 — Comptes et sessions**. Avant de coder : (a) le plan de fichiers par commit, (b) la liste des décisions de sécurité que tu prends et celles que tu me laisses (énumération d'e-mails, fournisseur mail, durée de session). Puis exécute. Tests d'auth exigés par `auth.md` au vert, `typecheck` et `lint` au vert.
