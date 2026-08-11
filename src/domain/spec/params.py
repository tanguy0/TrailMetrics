"""Declarative parameter schemas — the thing that generates plot forms.

A plot type never writes UI. It declares its parameters as :class:`ParamSpec`
objects, and a client turns that list into controls. This is what makes
"sub-parameters appear once a plot is added" fall out for free.

Everything here is **JSON-serializable, including the conditional logic**. That is
deliberate and load-bearing: the schema is served over HTTP and the browser
renders the same form, evaluates the same visibility rules and resolves the same
dynamic choice lists as Python does. A callable predicate would have forced the
frontend to re-implement each rule by hand and drift from the backend, so
visibility is expressed as a :class:`Condition` tree with evaluators on both sides.

Three pieces make it expressive enough for the real plots:

* ``visible_when`` — a condition over the values chosen so far, so a parameter
  only shows when it is relevant (XGBoost bin width only with that model).
* ``GROUP`` / ``LIST`` kinds — nested and *repeatable* parameters, which is how
  smoothing configs and named HR bands are expressed without special-casing.
* ``choices_from`` — options resolved against a registry, so adding an activity
  metric makes it selectable in every plot that takes a metric, with no edit here.

Schemas are code; only the **values** a user picks are persisted, which is why
:func:`coerce` fills defaults and drops unknown keys when loading a page saved by
an older version of the app.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ParamKind(str, Enum):
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    TEXT = "text"
    TEXTAREA = "textarea"        # multi-line text; a paragraph, not a label
    IMAGE = "image"              # an image URL, with an upload control beside it
    CHOICE = "choice"            # single value from ``choices``
    MULTICHOICE = "multichoice"  # list of values from ``choices``
    GROUP = "group"              # nested dict shaped by ``children``
    LIST = "list"                # list of dicts, each shaped by ``children``


@dataclass(frozen=True)
class Choice:
    """One selectable option. ``label_key`` is translated when the schema is served."""

    value: str
    label_key: str

    def to_dict(self, lang: str) -> Dict[str, Any]:
        from src.translations import translate
        return {"value": self.value, "label": translate(self.label_key, lang)}


# --- Conditions ------------------------------------------------------------

class ConditionOp(str, Enum):
    EQ = "eq"                    # values[key] == value
    NE = "ne"
    ONE_OF = "one_of"            # values[key] in value (a list)
    CONTAINS = "contains"        # values[key] (a list) contains value
    TRUTHY = "truthy"
    FALSY = "falsy"
    NONEMPTY = "nonempty"        # list/string with something in it
    EMPTY = "empty"
    # Registry-aware: true when the metric named by values[key] offers a choice
    # of aggregations (ratios and counts fix their own, so the control is hidden).
    METRIC_ALLOWS_AGG = "metric_allows_agg"
    ALL_OF = "all_of"
    ANY_OF = "any_of"
    NOT = "not"


@dataclass
class Condition:
    """A serializable predicate over the parameter values chosen so far."""

    op: ConditionOp
    key: Optional[str] = None
    value: Any = None
    conditions: List["Condition"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"op": self.op.value}
        if self.key is not None:
            payload["key"] = self.key
        if self.value is not None:
            payload["value"] = self.value
        if self.conditions:
            payload["conditions"] = [c.to_dict() for c in self.conditions]
        return payload

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "Condition":
        return Condition(
            op=ConditionOp(raw["op"]),
            key=raw.get("key"),
            value=raw.get("value"),
            conditions=[Condition.from_dict(c) for c in (raw.get("conditions") or [])],
        )


class when:
    """Builders for :class:`Condition`, so plot modules read as declarations."""

    @staticmethod
    def eq(key: str, value: Any) -> Condition:
        return Condition(op=ConditionOp.EQ, key=key, value=value)

    @staticmethod
    def ne(key: str, value: Any) -> Condition:
        return Condition(op=ConditionOp.NE, key=key, value=value)

    @staticmethod
    def one_of(key: str, values: List[Any]) -> Condition:
        return Condition(op=ConditionOp.ONE_OF, key=key, value=list(values))

    @staticmethod
    def contains(key: str, value: Any) -> Condition:
        return Condition(op=ConditionOp.CONTAINS, key=key, value=value)

    @staticmethod
    def truthy(key: str) -> Condition:
        return Condition(op=ConditionOp.TRUTHY, key=key)

    @staticmethod
    def falsy(key: str) -> Condition:
        return Condition(op=ConditionOp.FALSY, key=key)

    @staticmethod
    def nonempty(key: str) -> Condition:
        return Condition(op=ConditionOp.NONEMPTY, key=key)

    @staticmethod
    def empty(key: str) -> Condition:
        return Condition(op=ConditionOp.EMPTY, key=key)

    @staticmethod
    def metric_allows_agg(key: str = "metric") -> Condition:
        return Condition(op=ConditionOp.METRIC_ALLOWS_AGG, key=key)

    @staticmethod
    def all_of(*conditions: Condition) -> Condition:
        return Condition(op=ConditionOp.ALL_OF, conditions=list(conditions))

    @staticmethod
    def any_of(*conditions: Condition) -> Condition:
        return Condition(op=ConditionOp.ANY_OF, conditions=list(conditions))

    @staticmethod
    def not_(condition: Condition) -> Condition:
        return Condition(op=ConditionOp.NOT, conditions=[condition])


def evaluate(condition: Optional[Condition], values: Dict[str, Any]) -> bool:
    """Evaluate a condition against the current values; unknown ops pass."""
    if condition is None:
        return True

    op = condition.op
    if op is ConditionOp.ALL_OF:
        return all(evaluate(c, values) for c in condition.conditions)
    if op is ConditionOp.ANY_OF:
        return any(evaluate(c, values) for c in condition.conditions)
    if op is ConditionOp.NOT:
        return not all(evaluate(c, values) for c in condition.conditions)

    current = values.get(condition.key) if condition.key else None

    if op is ConditionOp.EQ:
        return current == condition.value
    if op is ConditionOp.NE:
        return current != condition.value
    if op is ConditionOp.ONE_OF:
        return current in (condition.value or [])
    if op is ConditionOp.CONTAINS:
        return condition.value in (current or [])
    if op is ConditionOp.TRUTHY:
        return bool(current)
    if op is ConditionOp.FALSY:
        return not bool(current)
    if op is ConditionOp.NONEMPTY:
        return bool(current)
    if op is ConditionOp.EMPTY:
        return not bool(current)
    if op is ConditionOp.METRIC_ALLOWS_AGG:
        # Imported lazily: the metric registry sits above this module. Checked
        # against FITNESS_FATIGUE_METRICS first — metric_trend is the one place
        # those keys are selectable, and they're deliberately absent from
        # ACTIVITY_METRICS (see that dict's docstring), so metric_or_default
        # alone would never find them and would fall back to the default metric.
        from src.domain.dataset.metrics import FITNESS_FATIGUE_METRICS, metric_or_default
        metric = FITNESS_FATIGUE_METRICS.get(current) or metric_or_default(current)
        return not metric.is_fixed_agg
    return True


# --- Parameter specs -------------------------------------------------------

@dataclass
class ParamSpec:
    """One parameter of a plot (or of a nested group)."""

    key: str
    kind: ParamKind
    label_key: str
    default: Any = None
    # Static options for CHOICE / MULTICHOICE.
    choices: List[Choice] = field(default_factory=list)
    # Dynamic options, resolved by the client against the registry payload. Known
    # providers: "activity_metrics", "aggregations", "granularities",
    # "pr_distances", "gradient_bands", "sortable_columns".
    choices_from: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    help_key: Optional[str] = None
    # GROUP / LIST: the schema of the nested values.
    children: List["ParamSpec"] = field(default_factory=list)
    # LIST: cap on rows the user may add.
    max_items: Optional[int] = None
    # Show this parameter only when the condition holds.
    visible_when: Optional[Condition] = None

    def to_dict(self, lang: str) -> Dict[str, Any]:
        """Wire format for the client's form generator."""
        from src.translations import translate
        payload: Dict[str, Any] = {
            "key": self.key,
            "kind": self.kind.value,
            "label": translate(self.label_key, lang),
            "default": self.default,
        }
        if self.choices:
            payload["choices"] = [c.to_dict(lang) for c in self.choices]
        if self.choices_from:
            payload["choices_from"] = self.choices_from
        for name in ("min", "max", "step", "max_items"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        if self.help_key:
            payload["help"] = translate(self.help_key, lang)
        if self.children:
            payload["children"] = [c.to_dict(lang) for c in self.children]
        if self.visible_when is not None:
            payload["visible_when"] = self.visible_when.to_dict()
        return payload


def defaults(specs: List[ParamSpec]) -> Dict[str, Any]:
    """The default value of every parameter in ``specs`` (recursively)."""
    out: Dict[str, Any] = {}
    for spec in specs:
        if spec.kind is ParamKind.GROUP:
            out[spec.key] = defaults(spec.children)
        elif spec.kind is ParamKind.LIST:
            out[spec.key] = [dict(row) for row in (spec.default or [])]
        else:
            out[spec.key] = spec.default
    return out


def coerce(specs: List[ParamSpec], values: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge stored ``values`` onto the schema's defaults.

    Missing keys get their default and unknown keys are dropped, so a page saved
    before a plot gained (or lost) a parameter still loads. Types are nudged into
    shape because a JSON round-trip loses ints vs floats.
    """
    values = values or {}
    out = defaults(specs)
    for spec in specs:
        if spec.key not in values:
            continue
        raw = values[spec.key]
        if spec.kind is ParamKind.GROUP:
            out[spec.key] = coerce(spec.children, raw if isinstance(raw, dict) else {})
        elif spec.kind is ParamKind.LIST:
            rows_in = raw if isinstance(raw, list) else []
            out[spec.key] = [
                coerce(spec.children, row) for row in rows_in if isinstance(row, dict)
            ]
        else:
            out[spec.key] = _coerce_scalar(spec, raw)
    return out


def _coerce_scalar(spec: ParamSpec, raw: Any) -> Any:
    if raw is None:
        return spec.default
    try:
        if spec.kind is ParamKind.BOOL:
            return bool(raw)
        if spec.kind is ParamKind.INT:
            return int(raw)
        if spec.kind is ParamKind.FLOAT:
            return float(raw)
        if spec.kind is ParamKind.MULTICHOICE:
            return list(raw) if isinstance(raw, (list, tuple)) else [raw]
        if spec.kind in (ParamKind.TEXT, ParamKind.TEXTAREA, ParamKind.IMAGE):
            return str(raw)
    except (TypeError, ValueError):
        return spec.default
    return raw


def visible(spec: ParamSpec, values: Dict[str, Any]) -> bool:
    """Whether ``spec`` should be shown given the values chosen so far."""
    return evaluate(spec.visible_when, values)


def find(specs: List[ParamSpec], key: str) -> Optional[ParamSpec]:
    for spec in specs:
        if spec.key == key:
            return spec
    return None


# --- Small builders, so plot modules read as declarations ------------------

def boolean(key: str, label_key: str, default: bool = False, **kw) -> ParamSpec:
    return ParamSpec(key=key, kind=ParamKind.BOOL, label_key=label_key,
                     default=default, **kw)


def integer(key: str, label_key: str, default: int, **kw) -> ParamSpec:
    return ParamSpec(key=key, kind=ParamKind.INT, label_key=label_key,
                     default=default, **kw)


def number(key: str, label_key: str, default: float, **kw) -> ParamSpec:
    return ParamSpec(key=key, kind=ParamKind.FLOAT, label_key=label_key,
                     default=default, **kw)


def text(key: str, label_key: str, default: str = "", **kw) -> ParamSpec:
    return ParamSpec(key=key, kind=ParamKind.TEXT, label_key=label_key,
                     default=default, **kw)


def textarea(key: str, label_key: str, default: str = "", **kw) -> ParamSpec:
    """Multi-line text. Same value as :func:`text`, a different control."""
    return ParamSpec(key=key, kind=ParamKind.TEXTAREA, label_key=label_key,
                     default=default, **kw)


def image(key: str, label_key: str, default: str = "", **kw) -> ParamSpec:
    """An image URL. The client pairs the field with an upload button."""
    return ParamSpec(key=key, kind=ParamKind.IMAGE, label_key=label_key,
                     default=default, **kw)


def choice(key: str, label_key: str, default: str, **kw) -> ParamSpec:
    return ParamSpec(key=key, kind=ParamKind.CHOICE, label_key=label_key,
                     default=default, **kw)


def multichoice(key: str, label_key: str, default: Optional[List[str]] = None,
                **kw) -> ParamSpec:
    return ParamSpec(key=key, kind=ParamKind.MULTICHOICE, label_key=label_key,
                     default=list(default or []), **kw)


def group(key: str, label_key: str, children: List[ParamSpec], **kw) -> ParamSpec:
    return ParamSpec(key=key, kind=ParamKind.GROUP, label_key=label_key,
                     children=children, **kw)


def rows(key: str, label_key: str, children: List[ParamSpec],
         default: Optional[List[dict]] = None, **kw) -> ParamSpec:
    return ParamSpec(key=key, kind=ParamKind.LIST, label_key=label_key,
                     children=children, default=list(default or []), **kw)
