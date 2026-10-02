# Comptes, sessions et sécurité (v2)

Objectif : e-mail + mot de passe, minimaliste à l'écran, sérieux dessous. L'identité cesse d'être l'athlète Strava.

## Modèle de données

```sql
create extension if not exists citext;

create table accounts (
    id             uuid primary key default gen_random_uuid(),
    email          citext unique not null,
    password_hash  text not null,                 -- argon2id, paramètres encodés dans le hash
    role           text not null default 'athlete' check (role in ('athlete','coach','master')),
    lang           text not null default 'en',
    email_verified_at timestamptz,
    created_at     timestamptz not null default now(),
    last_login_at  timestamptz
);

-- l'athlète Strava devient un attachement optionnel du compte
alter table athletes add column account_id uuid unique references accounts(id) on delete set null;

create table sessions (
    id          uuid primary key default gen_random_uuid(),
    account_id  uuid not null references accounts(id) on delete cascade,
    token_hash  bytea not null unique,            -- sha256 du jeton opaque, jamais le jeton
    created_at  timestamptz not null default now(),
    expires_at  timestamptz not null,
    last_seen_at timestamptz,
    user_agent  text
);

create table password_resets (
    token_hash  bytea primary key,
    account_id  uuid not null references accounts(id) on delete cascade,
    expires_at  timestamptz not null,
    used_at     timestamptz
);

create table login_attempts (                     -- limitation de débit
    key         text not null,                    -- 'email:<email>' ou 'ip:<ip>'
    window_start timestamptz not null,
    count       integer not null default 0,
    primary key (key, window_start)
);
```

Toutes les tables liées à l'athlète restent clés par `athlete_id` (Strava) : ce qui vient de Strava appartient à l'athlète, ce qui vient du compte (plans de course, zones estimées, demandes de coaching) se re-clé sur `account_id`. Migration : `race_plans` et les zones self-reported gagnent `account_id`, rempli depuis `athletes.account_id` une fois le rattachement fait.

## Rattachement Strava

Depuis un compte connecté, « Connecter Strava » lance l'OAuth existant. Au callback : si l'athlète Strava n'existe pas → créé avec `account_id` ; s'il existe sans compte (utilisateur historique) → rattaché ; s'il existe **avec un autre compte** → refus explicite (« Ce Strava est déjà lié à un autre compte TAGG »). Les utilisateurs actuels créent un compte puis connectent Strava : leurs données réapparaissent par le rattachement, rien n'est perdu. Déconnecter Strava retire les credentials, garde l'athlète et ses données attachées au compte.

## Mots de passe

- Hachage **argon2id** (`argon2-cffi`), paramètres par défaut de la lib (m=64 Mo, t=3, p=4), re-hachage à la connexion si les paramètres ont changé.
- Règles : 10 caractères minimum, 128 maximum, pas de règle de composition (elles affaiblissent), refus des 10 000 mots de passe les plus courants (liste embarquée). Pas d'appel réseau.
- Jamais de mot de passe dans les logs, les erreurs, les URL.

## Sessions

- Jeton **opaque** de 32 octets (`secrets.token_urlsafe`), stocké haché (sha256) en base ; le cookie `tm_session` porte le jeton en clair, `HttpOnly`, `Secure`, `SameSite=Lax`, `Path=/`, 30 jours glissants (`expires_at` repoussé à chaque requête espacée de plus d'une heure). Remplace le JWT actuel : une session doit pouvoir être révoquée (déconnexion, « déconnecter tous mes appareils », changement de mot de passe).
- `api/deps.py` : `current_account` lit le cookie → `sessions` → `accounts` ; `current_athlete_id` dérive de `athletes.account_id` (et du cookie `tm_view_as` pour un coach, inchangé). Le service token web→API reste.
- Connexion : réponse **identique** (« E-mail ou mot de passe incorrect », même temps de réponse approximatif via un hachage factice quand l'e-mail n'existe pas) ; rotation du jeton à chaque connexion ; `last_login_at`.
- Inscription : réponse identique que l'e-mail existe ou non (« Si cette adresse est libre, votre compte est créé ») — ou, plus simple pour un petit produit, dire « Cette adresse a déjà un compte » : acceptable, l'énumération d'e-mails est un risque mineur ici. **Choisir et documenter.**

## Limitation de débit

- 10 tentatives / 15 min par e-mail et 50 / 15 min par IP sur `/auth/login`, 5 / h par IP sur `/auth/register` et `/auth/reset`. Au-delà : 429 et le même message générique. Compteurs en base (`login_attempts`), suffisant à cette échelle ; pas de Redis.

## CSRF, en-têtes

- `SameSite=Lax` + vérification de l'en-tête `Origin` sur toute requête mutante côté web (`/api/*` Next) ; les formulaires d'auth sont des `POST` same-origin.
- En-têtes : `Strict-Transport-Security`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`, CSP au moins pour `script-src 'self'` + les CDN utilisés.
- Cookies d'auth jamais lus côté client (déjà le cas).

## Réinitialisation et vérification d'e-mail

Il n'y a pas d'envoi d'e-mail dans le projet aujourd'hui. Il en faut un pour la réinitialisation (sinon un mot de passe oublié = compte perdu). Proposition : **Resend** (ou SMTP) derrière une interface `MailSender` ; sans variable d'environnement, le bouton « Mot de passe oublié » affiche « Écrivez à {MASTER_EMAIL} » au lieu d'envoyer. Jeton de reset : 32 octets, haché en base, 30 min, usage unique, invalide toutes les sessions à l'usage. La vérification d'e-mail est **optionnelle** en v2 (rien de sensible n'est envoyé par e-mail) ; champ prévu, flux plus tard.

## Rôles

`role` remplace `COACH_ATHLETE_IDS` et `MASTER_EMAIL` : `master` = toi (blog + coach), `coach` = peut accepter des demandes et voir ses coachés, `athlete` = tout le monde. Migration : le compte dont l'e-mail = `MASTER_EMAIL` reçoit `master` à sa création.

## Pages web

- `/login`, `/register`, `/reset` (+ `/reset/[token]`) : composant `AuthCard` (voir design), même shell que le site (rail réduit au lockup).
- Middleware Next : session présente → `/` redirige vers `/home` ; absente → `/home`, `/analyses`, `/coaching` rendent leur `Teaser` (règle v1.1), `/tools/*` libre.
- Déconnexion : `POST /auth/logout` supprime la session et le cookie.

## Tests à exiger

Hachage/vérification, message d'erreur identique pour e-mail inconnu et mot de passe faux, expiration et rotation de session, 429 après le seuil, rattachement Strava dans les trois cas, qu'un coach ne voit que ses coachés, qu'un `athlete` ne peut pas appeler les routes coach.
