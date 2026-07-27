/**
 * Minimal declaration for `plotly.js-dist-min`, which ships no types.
 *
 * Only the three calls the renderer makes are declared. Pulling in
 * `@types/plotly.js` would drag the full typed API for a surface this small, and
 * the chart IR is where the real type safety lives anyway.
 */
declare module "plotly.js-dist-min" {
  interface PlotlyStatic {
    react(
      element: HTMLElement,
      data: unknown[],
      layout?: Record<string, unknown>,
      config?: Record<string, unknown>,
    ): Promise<unknown>;
    purge(element: HTMLElement): void;
    Plots: { resize(element: HTMLElement): void };
  }
  const Plotly: PlotlyStatic;
  export default Plotly;
}
