# Color system

The app's UI chrome (not chart data — see [Out of scope](#out-of-scope-the-chart-palette)
below) follows a small number of rules rather than a palette to pick from freely. If
you're about to write `background: #`, or invent a seventh named hue, stop — one of the
systems below almost certainly already covers it.

There are three independent systems, plus a handful of fixed roles. They share two
colors (gold and red) at their extremes, which is deliberate — it's one visual
language — but **the same color means three different things depending on which
system placed it there**. Don't assume "gold" always means the same thing; check which
system you're in.

## 1. The section-accent scale — position in an ordered sequence

**Meaning:** where a section sits in a page's top-to-bottom (or first-to-last) order.
Gold is first, red is last, sections in between interpolate.

**Tokens** (`web/app/globals.css` `:root`): `--scale-1` … `--scale-6`, each with a
`-rgb` and a `-tint` companion (`--scale-3-rgb`, `--scale-3-tint`, ...). `--scale-1` is
literally `--sunrise`; `--scale-6` is literally `--danger` — they're the same colors as
the fixed roles below, reused as this scale's endpoints.

**The link is automatic, not per-element.** A `.scale-N` utility class sets three
*inherited* custom properties on whatever it's applied to:

```css
.scale-3 {
  --section-accent: var(--scale-3);
  --section-accent-rgb: var(--scale-3-rgb);
  --section-accent-tint: var(--scale-3-tint);
}
```

Apply one `.scale-N` class to a section's root wrapper, and every descendant that reads
`var(--section-accent, ...)` picks it up through ordinary CSS inheritance — there is no
per-child modifier class to keep in sync. Consumers today: `.card-block__title`,
`.tile`, `.record`, `.hr-map__*`, `.step`, `.panel__title`, `.plot-card`. Always give
the read a neutral or `--primary` fallback (`var(--section-accent, var(--primary))`) so
an element outside any scaled section doesn't render broken.

**A page with a variable number of sections** (the Analysis page-builder, where a page
holds however many panels the user added) can't be covered by six fixed classes.
`web/lib/colorScale.ts` interpolates the identical two endpoints and tint formula in
JS — `scaleStepStyle(index, count)` returns a `style` object with the same three
`--section-accent*` properties, spread onto the element instead of a class. Keep the
two in sync if the endpoints ever change.

**A page with exactly one section** still follows the rule: it sits at position 0 of 1,
which is gold (`t=0`), not an arbitrary "default" color. This is why Analysis's "How an
analysis works" banner (`.explainer`, the only section on that page) carries `.scale-1`
rather than a fixed blue.

## 2. The sport-family scale — which sport, not where on the page

**Meaning:** which of the four sport families (running, hiking, cycling, swimming) an
activity or a total belongs to. Fixed order, fixed mapping, not positional.

**Tokens:** `--sport-scale-1` (running, gold) → `--sport-scale-2` (hiking) →
`--sport-scale-3` (cycling) → `--sport-scale-4` (swimming, red), each with `-rgb`/
`-tint`. Four evenly-spaced steps across the *same* gold-to-red range as §1 — but its
own token set, not a subset of `--scale-1..6`: four points split 0–1 differently than
six do.

**One function decides the mapping:** `sportTone()` in `web/lib/sport.ts`. It collapses
every Strava sport type onto one of five tones — `"running" | "hiking" | "cycling" |
"swimming" | "neutral"` — and running is one tone regardless of trail, road, or
virtual. Don't re-derive this elsewhere; import it.

**Consumers:** `.training-session--*` (the calendar's completed-session chip),
`.last-activity__sport--*` (Home's last-activity tag), `.week-summary--*` (a training
week's per-sport totals). All three read the tone the same way
(`` `week-summary--${tone}` ``, `` `.training-session--${sportTone(type)}` ``) so a
sport looks identical everywhere it appears. `neutral` (an unrecognized sport type)
always falls back to plain `--surface-alt`/`--spine`/`--muted`, not a scale step.

## 3. Green is the app's one button color

Every button in the app is green (`--primary`, which is the same hex as
`--tone-forest`). `.button`, `.button--ghost`, `.sidebar__link`, `.new-page`,
`.training-pill--goal` — all of it.

**Two deliberate, permanent exceptions** — don't "fix" these to be green:

- **`.button--danger`** stays red (`--danger`). Destructive actions keep the
  near-universal red safety convention; a green delete button removes the visual pause
  that color is there to create.
- **`.button--strava`** stays Strava's own brand orange (`#fc4c02`). Standard
  third-party-sign-in UX (the same reason a "Sign in with Google" button is
  Google-colored) — it has to read as *Strava's* action, not the app's.

**Adjacent, but not buttons — don't force these green either:**

- `.training-pill--workout` is blue (`--tone-blue`), tinted rather than solid, to keep
  the lighter visual weight a workout has always had next to a goal's solid fill.
- `.training-pill--note` is neutral grey. Notes are background information, not
  something to act on.
- The trend-direction badges (§4) use green for a *reason unrelated to buttons* — see
  below.

If you're adding a new button and reach for anything other than `.button`/
`.button--ghost`, stop and ask whether it should just be a plain button.

## 4. Trend badges — a fourth, unrelated meaning for the same three colors

`.trend-badge--increasing` (green, `--moss`), `.trend-badge--stable` (gold,
`--sunrise`), `.trend-badge--decreasing` (red, `--danger`) encode **the direction a
metric is moving**, on Home's Recent Efficiency/Recent Form badges. This is a fourth,
independent system that happens to reuse the same three colors as the extremes of §1/§2
and the button green of §3 — green here means "going up," not "click me" or "first in
the list." Don't assume badges, buttons, and the position scale all read the same way
just because a hex value matches.

## Every color needs its `-rgb` and `-tint` companions

Any token used as a background fill needs both:

- **`-tint`**: that color at 10% over `--surface`, so a light background stays legible
  — **precomputed, not `color-mix()`'d at render time**, so contrast is a fixed,
  checked number (body ink lands around 14:1, muted labels around 5.2:1) rather than
  something that could drift with a browser's color-space handling. Formula: `tint =
  0.9 × surface + 0.1 × accent`, per channel, with `surface = (255, 253, 249)` (`#fffdf9`,
  i.e. `--surface`).
- **`-rgb`**: the same color as a bare `"r, g, b"` triplet, for `rgba(var(--x-rgb),
  0.2)` — box-shadows, print-texture overlays, anything that needs an alpha CSS custom
  properties can't apply to a hex string directly.

When a token is already identical to another (`--primary` and `--tone-forest` are the
same hex), reuse the existing `-rgb`/`-tint` rather than defining a second name for the
same triplet — see `.training-pill--goal.training-pill--secondary`'s comment for why.

## Decision guide

- **Adding a new section to a page that already has an ordered sequence of them**
  (another Home card, another panel type)? Give it the next `.scale-N`, or thread it
  through `scaleStepStyle` if the page's section count is dynamic. Don't pick a hue.
- **Adding a new sport family**? There usually isn't one — the four are fixed by
  `RUNNING_SPORT_TYPES`/`CYCLING_SPORT_TYPES`/`HIKING_SPORT_TYPES`/`SWIMMING_SPORT_TYPES`
  in `web/lib/sport.ts` (mirrored in `src/domain/dataset/sport.py`). If a genuinely new
  family shows up, it needs a fifth `--sport-scale` step and the existing four
  re-spaced — that's a deliberate, visible change, not something to bolt on with an
  arbitrary sixth hue.
- **Adding a new button?** `.button` (+ `.button--ghost`/`.button--small`/
  `.button--wide` as needed). Only reach for `--danger` if the action is genuinely
  destructive, and never invent a new brand-colored button without the same
  third-party-sign-in justification `.button--strava` has.
- **Adding a status/direction indicator** (something is trending up, is stale, has
  failed)? That's the trend-badge pattern (§4) — green/gold/red for
  increasing/stable/decreasing — not the section scale or the sport scale, even though
  the colors overlap.
- **None of the above fits?** Reuse an existing named role
  (`--tone-forest`/`--tone-terracotta`/`--tone-plum`/`--tone-blue`/`--moss`/`--danger`/
  `--sunrise`) for what it already means elsewhere, rather than adding an eighth. If
  nothing fits, that's a real gap — raise it rather than picking a hex that looks nice
  next to the others.

## Out of scope: the chart palette

Plot/trace colors (the lines and bars inside a chart, as opposed to the page around it)
are a separate system: `src/domain/charts/plotly.py` / `plotting_common.py` on the
Python side, mirrored by hand in `web/lib/theme.ts` for the Leaflet route map and any
client-side color math (Plotly and Leaflet can't consume a CSS `var()`, so this mirror
has to stay a literal hex duplicate — update both sides together). It follows its own
rules (data encoding, palette-validator-checked color-blindness separation) and isn't
governed by anything in this document.
