"""The colour tokens live in four places; this keeps them from drifting.

    /path/to/venv/bin/python -m unittest discover -s tests -v

`design/tagg/tokens.json` is the source of truth. `src/domain/gap/theme.py`
(Plotly, server side), `web/lib/theme.ts` (Plotly and Leaflet, browser side) and
`web/app/tokens.css` (every stylesheet) each restate it, and must say exactly the
same thing.
"""

import json
import re
import unittest
from pathlib import Path

from src.domain.gap import theme

ROOT = Path(__file__).resolve().parents[1]
TOKENS_JSON = ROOT / "design" / "tagg" / "tokens.json"
THEME_TS = ROOT / "web" / "lib" / "theme.ts"
TOKENS_CSS = [ROOT / "web" / "app" / "tokens.css", ROOT / "design" / "tagg" / "tokens.css"]


def _normalize(value: str) -> str:
    return value.replace(" ", "").lower()


def _json_colors() -> dict[str, str]:
    raw = {t["name"]: t["value"] for t in json.loads(TOKENS_JSON.read_text())["color"]["tokens"]}

    def resolve(value: str) -> str:
        alias = re.fullmatch(r"\{([a-z0-9-]+)\}", value)
        return resolve(raw[alias.group(1)]) if alias else value

    return {name: _normalize(resolve(value)) for name, value in raw.items()}


COLORS = _json_colors()


class ThemeTokensTest(unittest.TestCase):
    def test_theme_py_matches_tokens_json(self) -> None:
        for name, value in COLORS.items():
            with self.subTest(token=name):
                constant = name.upper().replace("-", "_")
                self.assertEqual(_normalize(getattr(theme, constant)), value)

    def test_theme_ts_matches_tokens_json(self) -> None:
        body = re.search(r"export const tokens = \{(.*?)\} as const;", THEME_TS.read_text(), re.S)
        self.assertIsNotNone(body, "theme.ts must export a `tokens` object")
        entries = re.findall(r'^\s*"?([a-z0-9-]+)"?:\s*"([^"]+)",', body.group(1), re.M)
        self.assertEqual({k: _normalize(v) for k, v in entries}, COLORS)

    def test_tokens_css_matches_tokens_json(self) -> None:
        for path in TOKENS_CSS:
            with self.subTest(path=str(path.relative_to(ROOT))):
                root = re.search(r":root\s*\{(.*?)\}", path.read_text(), re.S).group(1)
                declared = dict(re.findall(r"--([a-z0-9-]+):\s*([^;]+);", root))

                def resolve(value: str) -> str:
                    alias = re.fullmatch(r"var\(--([a-z0-9-]+)\)", value.strip())
                    return resolve(declared[alias.group(1)]) if alias else value

                self.assertEqual(
                    {name: _normalize(resolve(declared[name])) for name in COLORS}, COLORS
                )

    def test_tokens_css_copies_are_identical(self) -> None:
        self.assertEqual(TOKENS_CSS[0].read_text(), TOKENS_CSS[1].read_text())


if __name__ == "__main__":
    unittest.main()
