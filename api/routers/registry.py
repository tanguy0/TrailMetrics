"""The plot catalogue.

Unauthenticated on purpose: it is static, per-language metadata describing what the
app can plot, and the web app needs it to render forms before anyone signs in.
"""

from fastapi import APIRouter, Depends, Response

from api.deps import language
from api.serialization import registry_payload
from src.translations import LANGUAGES, ui_strings

router = APIRouter(tags=["registry"])


@router.get("/registry")
def registry(response: Response, lang: str = Depends(language)) -> dict:
    """Every plot type, its parameter schema, the metrics and the choice lists.

    Changes only when the code does, so it is safe for clients to hold on to for a
    while.
    """
    response.headers["Cache-Control"] = "public, max-age=300"
    return registry_payload(lang)


@router.get("/ui-strings")
def ui_strings_endpoint(response: Response, lang: str = Depends(language)) -> dict:
    """The web app's own wording, translated.

    Unauthenticated for the same reason as the registry: it is static per-language
    metadata, and the sign-in screen needs it before there is a session.
    """
    response.headers["Cache-Control"] = "public, max-age=300"
    return {
        "lang": lang,
        "languages": LANGUAGES,
        "strings": ui_strings(lang),
    }
