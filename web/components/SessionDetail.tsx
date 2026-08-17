"use client";

/**
 * A completed session: its numbers, its route, and its pace/GAP/heart-rate traces.
 *
 * Shared between Home's latest-activity widget and the Training calendar's
 * click-to-open session view, so there is exactly one "what a session looks like"
 * implementation rather than two drifting apart.
 *
 * The chart under the map is not bespoke plotting code. It is an ordinary
 * `stream_evolution` panel posted to `/render/panel` — the same pipeline
 * `HomeScreen`'s trend charts already use, and the same plot type the Analysis
 * builder exposes — scoped to this one activity via `mode: "activities"`. Pace,
 * GAP pace and heart rate are just three entries in that plot's `signals` list:
 * pace and GAP land on the primary axis (same unit), heart rate takes the second.
 */

import { useEffect, useState } from "react";

import { ChartView } from "@/components/ChartView";
import { RouteMap } from "@/components/RouteMap";
import {
  createComment,
  deleteComment,
  getActivityRoute,
  listComments,
  renderPanel,
  updateComment,
} from "@/lib/api";
import { formatDate, formatHms, formatNumber, formatPace, formatSpeed } from "@/lib/format";
import { CYCLING_SPORT_TYPES, HIKING_SPORT_TYPES, SWIMMING_SPORT_TYPES, sportTone } from "@/lib/sport";
import type { Translate } from "@/lib/strings";
import type {
  ActivityCard,
  ActivityComment,
  ChartData,
  DataSourceSpec,
  PanelSpec,
  RouteResult,
} from "@/lib/types";

function singleActivitySource(activityId: number): DataSourceSpec {
  return {
    mode: "activities",
    activity_ids: [activityId],
    selection_label: "",
    windows: [],
    filters: { sport_types: [], min_distance_km: null, max_distance_km: null },
  };
}

function paceGapHrPanel(activityId: number, isCycling: boolean, dropGap: boolean): PanelSpec {
  return {
    id: `panel_session_pace_${activityId}`,
    title: "",
    description: "",
    columns: 1,
    source: singleActivitySource(activityId),
    plots: [
      {
        id: `plot_session_pace_${activityId}`,
        plot_type: "stream_evolution",
        title: null,
        // GAP is a running-biomechanics model and means nothing for a bike, a
        // hike or a swim, so all three drop it; cycling additionally reads
        // its pace signal as speed instead.
        params: isCycling
          ? { signals: ["pace", "heartrate"], as_speed: true }
          : dropGap
            ? { signals: ["pace", "heartrate"] }
            : { signals: ["gap_pace", "pace", "heartrate"] },
      },
    ],
  };
}

export function SessionDetail({
  activity,
  t,
}: {
  activity: ActivityCard;
  t: Translate;
}) {
  const activityId = activity.activity_id;
  const [route, setRoute] = useState<RouteResult | null>(null);
  const [charts, setCharts] = useState<ChartData[] | null>(null);

  // The route is its own request: it can call out to Strava for an activity
  // imported before routes were stored, and the metrics above must not wait on it.
  useEffect(() => {
    let live = true;
    setRoute(null);
    getActivityRoute(activityId)
      .then((result) => live && setRoute(result))
      .catch(
        () => live && setRoute({ activity_id: activityId, points: [], source: "unavailable" }),
      );
    return () => {
      live = false;
    };
  }, [activityId]);

  const isCycling = CYCLING_SPORT_TYPES.includes(activity.sport_type);
  const dropGap = HIKING_SPORT_TYPES.includes(activity.sport_type)
    || SWIMMING_SPORT_TYPES.includes(activity.sport_type);

  useEffect(() => {
    let live = true;
    setCharts(null);
    renderPanel(paceGapHrPanel(activityId, isCycling, dropGap))
      .then((result) => {
        if (!live) return;
        setCharts(result.panel.plots.flatMap((plot) => plot.output?.charts ?? []));
      })
      // A missing-streams activity (a manual entry) leaves the rest of the view
      // intact; there is simply nothing to trace.
      .catch(() => live && setCharts([]));
    return () => {
      live = false;
    };
  }, [activityId, isCycling, dropGap]);

  const km = activity.distance_m != null ? activity.distance_m / 1000 : null;
  const pace = km && km > 0 && activity.moving_s != null ? activity.moving_s / km : null;
  const speedKmh = activity.moving_s && activity.moving_s > 0 && activity.distance_m != null
    ? (activity.distance_m / activity.moving_s) * 3.6 : null;

  return (
    <div className="session-detail">
      <p className="last-activity__head">
        <span className={`last-activity__sport last-activity__sport--${sportTone(activity.sport_type)}`}>
          {activity.sport_type}
        </span>
        <span className="last-activity__date">{formatDate(activity.date)}</span>
      </p>

      <CommentsSection activityId={activityId} t={t} />

      <dl className="metric-row">
        <Metric
          label={t("home.last.distance")}
          value={km != null ? `${formatNumber(km, 2)} ${t("common.km")}` : "—"}
        />
        <Metric label={t("home.last.time")} value={formatHms(activity.moving_s)} />
        {isCycling ? (
          <Metric label={t("home.last.speed")} value={formatSpeed(speedKmh)} />
        ) : (
          <Metric label={t("home.last.pace")} value={formatPace(pace)} />
        )}
        <Metric
          label={t("home.last.climb")}
          value={
            activity.elevation_gain_m != null
              ? `${formatNumber(activity.elevation_gain_m, 0)} ${t("common.metres")}`
              : "—"
          }
        />
        {activity.avg_hr != null && (
          <Metric
            label={t("home.last.heart_rate")}
            value={`${formatNumber(activity.avg_hr, 0)} bpm`}
          />
        )}
        {activity.avg_power_w != null && (
          <Metric
            label={t("home.last.power")}
            value={`${formatNumber(activity.avg_power_w, 0)} W`}
          />
        )}
      </dl>

      {route === null ? (
        <div className="pending">
          <span className="spinner" />
          <p className="muted">{t("home.last.map_loading")}</p>
        </div>
      ) : route.points.length ? (
        <RouteMap points={route.points} />
      ) : (
        <p className="muted">
          {route.source === "unavailable"
            ? t("home.last.map_unavailable")
            : t("home.last.map_none")}
        </p>
      )}

      {charts === null ? (
        <div className="pending">
          <span className="spinner" />
          <p className="muted">{t("common.loading")}</p>
        </div>
      ) : (
        charts.map((chart, index) => (
          <div className="chart-frame" key={index}>
            <ChartView chart={chart} />
          </div>
        ))
      )}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <dt className="metric__label">{label}</dt>
      <dd className="metric__value">{value}</dd>
    </div>
  );
}

function CommentsSection({ activityId, t }: { activityId: number; t: Translate }) {
  const [comments, setComments] = useState<ActivityComment[] | null>(null);
  const [draft, setDraft] = useState("");
  const [posting, setPosting] = useState(false);

  useEffect(() => {
    let live = true;
    setComments(null);
    listComments(activityId)
      .then((result) => live && setComments(result.comments))
      .catch(() => live && setComments([]));
    return () => {
      live = false;
    };
  }, [activityId]);

  const handleAdd = async () => {
    const body = draft.trim();
    if (!body) return;
    setPosting(true);
    try {
      const created = await createComment(activityId, body);
      setComments((current) => [...(current ?? []), created]);
      setDraft("");
    } finally {
      setPosting(false);
    }
  };

  return (
    <div className="session-comments">
      {comments?.map((comment) => (
        <CommentRow
          key={comment.id}
          activityId={activityId}
          comment={comment}
          t={t}
          onChange={(updated) =>
            setComments((current) =>
              (current ?? []).map((c) => (c.id === updated.id ? updated : c)),
            )
          }
          onDelete={() =>
            setComments((current) => (current ?? []).filter((c) => c.id !== comment.id))
          }
        />
      ))}
      <div className="session-comments__form">
        <textarea
          className="session-comments__input"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder={t("session.comments.placeholder")}
          rows={2}
        />
        <button
          type="button"
          className="button button--ghost button--small"
          disabled={posting || !draft.trim()}
          onClick={handleAdd}
        >
          {t("session.comments.add")}
        </button>
      </div>
    </div>
  );
}

function CommentRow({
  activityId,
  comment,
  t,
  onChange,
  onDelete,
}: {
  activityId: number;
  comment: ActivityComment;
  t: Translate;
  onChange: (updated: ActivityComment) => void;
  onDelete: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(comment.body);
  const [busy, setBusy] = useState(false);

  if (editing) {
    return (
      <div className="session-comments__form">
        <textarea
          className="session-comments__input"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          rows={2}
          autoFocus
        />
        <div className="session-comments__actions">
          <button
            type="button"
            className="button button--ghost button--small"
            disabled={busy}
            onClick={() => {
              setDraft(comment.body);
              setEditing(false);
            }}
          >
            {t("session.comments.cancel")}
          </button>
          <button
            type="button"
            className="button button--ghost button--small"
            disabled={busy || !draft.trim()}
            onClick={async () => {
              setBusy(true);
              try {
                onChange(await updateComment(activityId, comment.id, draft.trim()));
                setEditing(false);
              } finally {
                setBusy(false);
              }
            }}
          >
            {t("session.comments.save")}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="session-comments__item">
      <p className="session-comments__body">{comment.body}</p>
      <div className="session-comments__actions">
        <button
          type="button"
          className="button button--ghost button--small"
          onClick={() => setEditing(true)}
        >
          {t("session.comments.edit")}
        </button>
        <button
          type="button"
          className="button button--danger button--small"
          onClick={() => deleteComment(activityId, comment.id).then(onDelete)}
        >
          {t("session.comments.delete")}
        </button>
      </div>
    </div>
  );
}
