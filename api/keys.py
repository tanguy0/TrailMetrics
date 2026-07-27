"""Generate the secrets a deployment needs.

    python -m api.keys

Prints a fresh SESSION_SECRET, SERVICE_TOKEN and ENCRYPTION_KEY. Run it once per
environment and paste the values into the platform's variables — they should never
be committed, and the two sides of SERVICE_TOKEN must match.
"""

from api.security import generate_keys


def main() -> None:
    print("# Generated secrets — set these on the API, and SERVICE_TOKEN on the web app too.")
    for name, value in generate_keys().items():
        print(f"{name}={value}")
    print()
    print("# ENCRYPTION_KEY encrypts stored Strava tokens. Changing it does not lose")
    print("# any activity data, but every athlete has to reconnect Strava.")


if __name__ == "__main__":
    main()
