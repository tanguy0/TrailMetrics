"use client";

/**
 * An activity's route on an OpenStreetMap base layer.
 *
 * Leaflet is driven imperatively from an effect and loaded on demand — the same
 * approach `ChartView` takes with Plotly, and for the same reason: the library and
 * its CSS are dead weight on any screen without a map.
 *
 * Worth knowing: rendering this map makes the browser request tiles from the tile
 * host, which discloses the area the route covers to a third party. That is
 * inherent to a tiled map rather than something this component adds, but it is the
 * reason the component draws nothing at all when there is no route.
 */

import { useEffect, useRef, useState } from "react";

import { theme } from "@/lib/theme";

/**
 * Where the base tiles come from, and how we identify ourselves to that host.
 *
 * OSM's tile usage policy requires every request to identify the application, by
 * Referer or User-Agent; from a browser only the Referer is ours to set. This app
 * sends `Referrer-Policy: no-referrer` on every response (see next.config.mjs), so
 * tile requests arrived anonymous and osm.org answered them with its "tile usage
 * policy" placeholder image instead of a map. Hence `referrerPolicy` on the layer
 * below: an element's own referrer policy overrides the document's, so tile
 * requests — and only tile requests — carry this app's origin. `strict-origin-…`
 * rather than a full URL: the host needs to know which app is asking, not which
 * activity is being looked at.
 *
 * The URL is configurable because osm.org's servers are volunteer-run and the same
 * policy rules out apps with real traffic. Point NEXT_PUBLIC_MAP_TILE_URL (and the
 * matching attribution) at a provider with a paid plan once this app is past
 * hobby-scale; nothing else here has to change.
 */
const TILE_URL =
  process.env.NEXT_PUBLIC_MAP_TILE_URL || "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
const TILE_ATTRIBUTION =
  process.env.NEXT_PUBLIC_MAP_TILE_ATTRIBUTION ||
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

// Resolved once per session. The stylesheet is a side-effect import: Leaflet
// positions tiles with it, so the map is unusable without it.
let leafletPromise: Promise<typeof import("leaflet")> | null = null;
function loadLeaflet(): Promise<typeof import("leaflet")> {
  leafletPromise ??= Promise.all([
    import("leaflet"),
    // @ts-expect-error -- a CSS module has no type declarations; this is for its
    // side effect only, and the value is discarded below.
    import("leaflet/dist/leaflet.css"),
  ]).then(([module]) => module.default ?? module);
  return leafletPromise;
}

export function RouteMap({
  points,
  height = 260,
}: {
  /** `[latitude, longitude]` pairs, oldest first. */
  points: [number, number][];
  height?: number;
}) {
  const container = useRef<HTMLDivElement | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!container.current || points.length === 0) return;

    let map: import("leaflet").Map | null = null;
    let cancelled = false;

    loadLeaflet()
      .then((L) => {
        if (cancelled || !container.current) return;

        map = L.map(container.current, {
          // A route is a static picture here; scroll-zoom would hijack the page.
          scrollWheelZoom: false,
          zoomControl: true,
          attributionControl: true,
        });

        L.tileLayer(TILE_URL, {
          maxZoom: 18,
          attribution: TILE_ATTRIBUTION,
          referrerPolicy: "strict-origin-when-cross-origin",
        }).addTo(map);

        const line = L.polyline(points, {
          color: theme.terra,
          weight: 4,
          opacity: 0.9,
          lineJoin: "round",
        }).addTo(map);

        // Start and finish, so the direction of travel is readable.
        const dot = (at: [number, number], color: string, label: string) =>
          L.circleMarker(at, {
            radius: 5,
            color: theme.bgSurface,
            weight: 2,
            fillColor: color,
            fillOpacity: 1,
          })
            .addTo(map!)
            .bindTooltip(label);
        dot(points[0], theme.forest, "Start");
        if (points.length > 1) dot(points[points.length - 1], theme.danger, "Finish");

        map.fitBounds(line.getBounds(), { padding: [16, 16] });
      })
      .catch(() => !cancelled && setFailed(true));

    return () => {
      cancelled = true;
      // Leaflet leaks handlers and a resize observer if the map outlives its node.
      map?.remove();
    };
  }, [points]);

  if (points.length === 0 || failed) return null;

  return (
    <div
      className="route-map"
      ref={container}
      style={{ height }}
      // The route is decoration beside the numbers, which carry the same facts.
      role="presentation"
    />
  );
}
