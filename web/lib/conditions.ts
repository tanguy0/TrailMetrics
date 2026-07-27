/**
 * Evaluator for `ParamSpec.visible_when`.
 *
 * The mirror of `evaluate()` in `src/domain/spec/params.py`. Both sides read the
 * same serialized condition tree, which is why a plot's conditional parameters
 * behave identically here and on the server without either one hard-coding a rule.
 *
 * Unknown operators return `true`: a client that hasn't caught up with a new
 * operator should show a parameter it can't reason about rather than hide it.
 */

import type { Condition, MetricInfo, ParamSpec } from "./types";

export interface ConditionContext {
  /** Metric table from /registry — needed by the `metric_allows_agg` operator. */
  metrics: Record<string, MetricInfo>;
}

export function evaluateCondition(
  condition: Condition | undefined,
  values: Record<string, unknown>,
  context: ConditionContext,
): boolean {
  if (!condition) return true;

  const children = condition.conditions ?? [];
  switch (condition.op) {
    case "all_of":
      return children.every((c) => evaluateCondition(c, values, context));
    case "any_of":
      return children.some((c) => evaluateCondition(c, values, context));
    case "not":
      return !children.every((c) => evaluateCondition(c, values, context));
  }

  const current = condition.key ? values[condition.key] : undefined;

  switch (condition.op) {
    case "eq":
      return current === condition.value;
    case "ne":
      return current !== condition.value;
    case "one_of":
      return Array.isArray(condition.value) && condition.value.includes(current);
    case "contains":
      return Array.isArray(current) && current.includes(condition.value);
    case "truthy":
      return Boolean(current);
    case "falsy":
      return !current;
    case "nonempty":
      return Array.isArray(current) ? current.length > 0 : Boolean(current);
    case "empty":
      return Array.isArray(current) ? current.length === 0 : !current;
    case "metric_allows_agg": {
      const metric = context.metrics[String(current ?? "")];
      // Absent metric: show the control rather than silently hiding an option.
      return metric ? metric.allowed_aggs.length > 0 : true;
    }
    default:
      return true;
  }
}

/** The parameters to show, given what has been chosen so far. */
export function visibleParams(
  specs: ParamSpec[],
  values: Record<string, unknown>,
  context: ConditionContext,
): ParamSpec[] {
  return specs.filter((spec) => evaluateCondition(spec.visible_when, values, context));
}
