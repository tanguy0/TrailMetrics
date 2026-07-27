"""The plot catalogue.

Unauthenticated on purpose: it is static, per-language metadata describing what the
app can plot, and the web app needs it to render forms before anyone signs in.
"""

from fastapi import APIRouter, Depends, Response

from api.deps import language
from api.serialization import registry_payload

router = APIRouter(tags=["registry"])


@router.get("/registry")
def registry(response: Response, lang: str = Depends(language)) -> dict:
    """Every plot type, its parameter schema, the metrics and the choice lists.

    Changes only when the code does, so it is safe for clients to hold on to for a
    while.
    """
    response.headers["Cache-Control"] = "public, max-age=300"
    return registry_payload(lang)
