"""Account operations from the command line: roles, and resets without mail.

    python -m api.roles set someone@example.com coach
    python -m api.roles show someone@example.com
    python -m api.roles reset-link someone@example.com

Roles live in ``accounts.role`` (design/specs/auth.md § Rôles). ``MASTER_EMAIL``
is promoted to ``master`` automatically once it verifies its address; this is
for everything else — making a coach, or the master by hand when no mail
provider is configured to send the verification link. Run where the API's
``DATABASE_URL`` is set (e.g. a ``railway ssh`` shell in the API container).

``reset-link`` prints the link a reset email would have carried (30 minutes,
single use, signs out every device) — for answering "I forgot my password"
while the deployment runs without mail. Check who is asking before sending it.
"""

import sys

from api.config import get_settings
from api.deps import get_account_repository, get_database
from api.routers.auth import RESET_TTL_S
from api.security import hash_token, new_token
from src.domain.ports.accounts import ROLES

USAGE = (
    "usage: python -m api.roles set <email> <athlete|coach|master>"
    " | show <email> | reset-link <email>"
)


def main(argv: list) -> int:
    if len(argv) < 2 or argv[0] not in ("set", "show", "reset-link"):
        print(USAGE)
        return 2
    accounts = get_account_repository()
    found = accounts.by_email(argv[1])
    if found is None:
        print(f"no account for {argv[1]}")
        return 1
    account = found[0]
    if argv[0] == "reset-link":
        token = new_token()
        accounts.create_reset(hash_token(token), account.id, RESET_TTL_S)
        print(f"{get_settings().web_app_url.rstrip('/')}/reset/{token}")
        print("valid 30 minutes, single use; following it signs out every device")
        return 0
    if argv[0] == "set":
        if len(argv) != 3 or argv[2] not in ROLES:
            print(USAGE)
            return 2
        accounts.set_role(account.id, argv[2])
        account.role = argv[2]
    verified = "verified" if account.email_verified else "not verified"
    print(f"{account.email}: {account.role} ({verified})")
    return 0


if __name__ == "__main__":
    try:
        code = main(sys.argv[1:])
    finally:
        get_database().close()
    sys.exit(code)
