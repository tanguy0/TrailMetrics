# Coaching — demande, acceptation, accès (v2)

## Données

```sql
create table coaching_requests (
    id          uuid primary key default gen_random_uuid(),
    account_id  uuid not null references accounts(id) on delete cascade,
    message     text not null default '',
    phone       text,                               -- E.164 si fourni
    contact     text not null check (contact in ('email','phone')),
    status      text not null default 'pending' check (status in ('pending','accepted','declined','withdrawn')),
    created_at  timestamptz not null default now(),
    decided_at  timestamptz,
    decided_by  uuid references accounts(id)
);
create unique index on coaching_requests (account_id) where status = 'pending';

create table coaching (
    coach_id    uuid not null references accounts(id) on delete cascade,
    athlete_id  uuid not null references accounts(id) on delete cascade,
    since       timestamptz not null default now(),
    primary key (coach_id, athlete_id)
);
```

## Règles

- Un compte n'a qu'une demande en attente ; il peut la modifier ou la retirer tant qu'elle l'est.
- Le formulaire exige **un** moyen de contact : l'e-mail est pré-rempli et coché ; si l'athlète préfère le téléphone, il le saisit et bascule le choix. Pas de validation stricte du numéro au-delà de « ressemble à un numéro » ; stocké tel quel + normalisé E.164 si possible.
- Accepter (`POST /coaching/requests/{id}/accept`, rôle coach ou master) : statut `accepted`, ligne dans `coaching`, et l'athlète voit la page Coaching à sa prochaine requête. Décliner : statut `declined`, l'athlète voit « Le coach ne peut pas vous prendre pour le moment » et peut re-demander après 30 jours.
- Le coach reçoit une **notification par e-mail** à chaque nouvelle demande si `MailSender` est configuré ; sinon la liste suffit (tu es seul coach).
- `is_coached(account)` = existe une ligne `coaching` où `athlete_id = account.id`. C'est ce prédicat qui ouvre la page, pas une variable d'environnement.
- Le sélecteur « Athlète » du rail (coach) liste `coaching.athlete_id` pour le coach connecté ; `tm_view_as` continue de porter l'athlète Strava courant.

## Écrans

- Non coaché : la page-offre décrite dans `access.md` (hero, trois points, aperçu du carnet en fond, formulaire). États : *à remplir*, *envoyée le …* (modifiable / retirer), *déclinée*.
- Coaché : le carnet actuel.
- Coach : comme un athlète coaché — son propre carnet ; le carnet de l'athlète sélectionné quand il en consulte un (bouton « Mes athlètes » pour revenir). Les demandes en attente arrivent en **notification dans le rail**, sous le sélecteur « Athlète » : un lien « Demandes de coaching » avec leur nombre (`tm-rail__count`), absent à zéro, qui ouvre la liste (accepter / décliner). Les athlètes coachés s'ouvrent depuis le sélecteur.
