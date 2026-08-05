"use client";

/**
 * Generates a plot's form from its parameter schema.
 *
 * There is no per-plot UI anywhere in this app. A plot type declares parameters in
 * Python, `/registry` serializes the schema, and this component renders it —
 * including nested groups, repeatable rows and conditional visibility. Adding a
 * plot type or a parameter is a backend-only change.
 */

import { useCallback, useState } from "react";

import { uploadAsset } from "@/lib/api";
import { evaluateCondition } from "@/lib/conditions";
import type { Choice, MetricInfo, ParamSpec } from "@/lib/types";

type Values = Record<string, unknown>;

interface FormProps {
  specs: ParamSpec[];
  values: Values;
  onChange: (values: Values) => void;
  providers: Record<string, Choice[]>;
  metrics: Record<string, MetricInfo>;
  idPrefix: string;
}

export function ParamForm({
  specs,
  values,
  onChange,
  providers,
  metrics,
  idPrefix,
}: FormProps) {
  const set = useCallback(
    (key: string, value: unknown) => onChange({ ...values, [key]: value }),
    [onChange, values],
  );

  const visible = specs.filter((spec) =>
    evaluateCondition(spec.visible_when, values, { metrics }),
  );

  return (
    <div className="param-form">
      {visible.map((spec) => (
        <ParamField
          key={spec.key}
          spec={spec}
          value={values[spec.key]}
          onChange={(next) => set(spec.key, next)}
          providers={providers}
          metrics={metrics}
          allValues={values}
          idPrefix={idPrefix}
        />
      ))}
    </div>
  );
}

interface FieldProps {
  spec: ParamSpec;
  value: unknown;
  onChange: (value: unknown) => void;
  providers: Record<string, Choice[]>;
  metrics: Record<string, MetricInfo>;
  allValues: Values;
  idPrefix: string;
}

function ParamField(props: FieldProps) {
  const { spec, value, onChange, providers, metrics, allValues, idPrefix } = props;
  const id = `${idPrefix}-${spec.key}`;
  const choices = resolveChoices(spec, providers, metrics, allValues);

  // --- Nested shapes ------------------------------------------------------

  if (spec.kind === "group") {
    const nested = (value ?? {}) as Values;
    return (
      <details className="param param--group">
        <summary>{spec.label}</summary>
        {spec.help && <p className="muted">{spec.help}</p>}
        <ParamForm
          specs={spec.children ?? []}
          values={nested}
          onChange={onChange}
          providers={providers}
          metrics={metrics}
          idPrefix={id}
        />
      </details>
    );
  }

  if (spec.kind === "list") {
    return (
      <ParamRows
        spec={spec}
        rows={(Array.isArray(value) ? value : []) as Values[]}
        onChange={onChange}
        providers={providers}
        metrics={metrics}
        idPrefix={id}
      />
    );
  }

  // --- Scalars ------------------------------------------------------------

  return (
    <div className="param">
      <label className="param__label" htmlFor={id}>
        {spec.label}
      </label>

      {spec.kind === "bool" && (
        <input
          id={id}
          type="checkbox"
          checked={Boolean(value)}
          onChange={(event) => onChange(event.target.checked)}
        />
      )}

      {(spec.kind === "int" || spec.kind === "float") && (
        <input
          id={id}
          type="number"
          value={value == null ? "" : String(value)}
          min={spec.min}
          max={spec.max}
          step={spec.step ?? (spec.kind === "int" ? 1 : "any")}
          onChange={(event) => {
            const raw = event.target.value;
            if (raw === "") return onChange(null);
            const parsed = spec.kind === "int" ? parseInt(raw, 10) : parseFloat(raw);
            onChange(Number.isNaN(parsed) ? null : parsed);
          }}
        />
      )}

      {spec.kind === "text" && (
        <input
          id={id}
          type="text"
          value={value == null ? "" : String(value)}
          onChange={(event) => onChange(event.target.value)}
        />
      )}

      {spec.kind === "textarea" && (
        <textarea
          id={id}
          className="param__textarea"
          rows={4}
          value={value == null ? "" : String(value)}
          onChange={(event) => onChange(event.target.value)}
        />
      )}

      {spec.kind === "image" && (
        <ImageField
          id={id}
          value={value == null ? "" : String(value)}
          onChange={onChange}
        />
      )}

      {spec.kind === "choice" && (
        <select
          id={id}
          value={value == null ? "" : String(value)}
          onChange={(event) => onChange(event.target.value)}
        >
          {choices.map((choice) => (
            <option key={choice.value} value={choice.value}>
              {choice.label}
            </option>
          ))}
        </select>
      )}

      {spec.kind === "multichoice" && (
        <MultiChoice
          id={id}
          choices={choices}
          selected={(Array.isArray(value) ? value : []) as string[]}
          onChange={onChange}
        />
      )}

      {spec.help && <p className="param__help">{spec.help}</p>}
    </div>
  );
}

/**
 * An image parameter: upload a file, or point at a URL.
 *
 * Both write the same thing — a URL string — so the stored parameter has one shape
 * and the plot type never learns that uploading exists. An upload just happens to
 * produce a URL served by this app.
 */
function ImageField({
  id,
  value,
  onChange,
}: {
  id: string;
  value: string;
  onChange: (value: unknown) => void;
}) {
  const [uploading, setUploading] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);

  const upload = async (file: File | undefined) => {
    if (!file) return;
    setUploading(true);
    setFailure(null);
    try {
      const asset = await uploadAsset(file);
      onChange(asset.url);
    } catch (error) {
      // The server's message is the useful one — it names the real limit or the
      // rejected type rather than "upload failed".
      setFailure((error as Error).message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="image-field">
      <div className="image-field__actions">
        <label className="button button--ghost button--small">
          {uploading ? "Uploading…" : "Upload"}
          {/* The file input itself is hidden: a <label>-wrapped input styles as a
              button, where a bare one cannot be. */}
          <input
            className="image-field__file"
            type="file"
            accept="image/png,image/jpeg,image/webp,image/gif"
            disabled={uploading}
            onChange={(event) => {
              upload(event.target.files?.[0]);
              // Cleared so re-picking the same file fires `change` again.
              event.target.value = "";
            }}
          />
        </label>
        {value && (
          <button
            type="button"
            className="button button--ghost button--small"
            onClick={() => onChange("")}
          >
            Remove
          </button>
        )}
      </div>

      <input
        id={id}
        type="text"
        placeholder="…or paste an image URL"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />

      {failure && <p className="note note--error">{failure}</p>}
      {value && (
        // A thumbnail here rather than only in the output: an image that fails to
        // load says so while the URL is still in front of you.
        <img className="image-field__preview" src={value} alt="" />
      )}
    </div>
  );
}

/** Checkbox list rather than a multi-select: far easier to use with many options. */
function MultiChoice({
  id,
  choices,
  selected,
  onChange,
}: {
  id: string;
  choices: Choice[];
  selected: string[];
  onChange: (value: string[]) => void;
}) {
  const toggle = (value: string) =>
    onChange(
      selected.includes(value)
        ? selected.filter((v) => v !== value)
        : // Keep the schema's option order, not click order, so legends stay stable.
          choices.map((c) => c.value).filter((v) => v === value || selected.includes(v)),
    );

  return (
    <div className="multichoice" id={id}>
      {choices.map((choice) => (
        <label key={choice.value} className="multichoice__item">
          <input
            type="checkbox"
            checked={selected.includes(choice.value)}
            onChange={() => toggle(choice.value)}
          />
          <span>{choice.label}</span>
        </label>
      ))}
    </div>
  );
}

/** Repeatable rows — how named HR bands are edited. */
function ParamRows({
  spec,
  rows,
  onChange,
  providers,
  metrics,
  idPrefix,
}: {
  spec: ParamSpec;
  rows: Values[];
  onChange: (value: Values[]) => void;
  providers: Record<string, Choice[]>;
  metrics: Record<string, MetricInfo>;
  idPrefix: string;
}) {
  const children = spec.children ?? [];
  const atLimit = spec.max_items != null && rows.length >= spec.max_items;

  const addRow = () => {
    const blank: Values = {};
    for (const child of children) blank[child.key] = child.default;
    onChange([...rows, blank]);
  };

  const updateRow = (index: number, next: Values) =>
    onChange(rows.map((row, i) => (i === index ? next : row)));

  return (
    <div className="param param--rows">
      <span className="param__label">{spec.label}</span>
      {spec.help && <p className="param__help">{spec.help}</p>}

      {rows.map((row, index) => (
        <div key={index} className="row-item">
          <ParamForm
            specs={children}
            values={row}
            onChange={(next) => updateRow(index, next)}
            providers={providers}
            metrics={metrics}
            idPrefix={`${idPrefix}-${index}`}
          />
          <button
            type="button"
            className="button button--ghost button--small"
            onClick={() => onChange(rows.filter((_, i) => i !== index))}
            aria-label="Remove row"
          >
            Remove
          </button>
        </div>
      ))}

      <button
        type="button"
        className="button button--ghost button--small"
        onClick={addRow}
        disabled={atLimit}
        title={atLimit ? `At most ${spec.max_items}` : undefined}
      >
        Add
      </button>
    </div>
  );
}

/**
 * Options for a parameter: static, or from a named provider.
 *
 * The one context-sensitive case is `aggregations`, narrowed to what the selected
 * metric actually allows — driven by the metric table from `/registry`, not by any
 * rule written here.
 */
function resolveChoices(
  spec: ParamSpec,
  providers: Record<string, Choice[]>,
  metrics: Record<string, MetricInfo>,
  values: Values,
): Choice[] {
  if (spec.choices?.length) return spec.choices;
  if (!spec.choices_from) return [];

  const provided = providers[spec.choices_from] ?? [];
  if (spec.choices_from !== "aggregations") return provided;

  const metric = metrics[String(values.metric ?? "")];
  if (!metric || !metric.allowed_aggs.length) return provided;
  return provided.filter((choice) => metric.allowed_aggs.includes(choice.value));
}
