# Design direction: The Timetable

Direction contract for every public surface. Change it here first, then change the surface.

## Audience and the one action

On-call engineers and homelabbers who already run Prometheus and see Datadog or Grafana Cloud forecasts every day. The action on every surface is to run `promcast scan` against their own Prometheus and read one board. A surface fails if it does not lead to that board.

## What it steals

Étienne-Jules Marey's graphical train timetable, Paris to Lyon, 1885 (the plate in Tufte, *The Visual Display of Quantitative Information*, p. 31). Stations are rules spaced by true distance, the clock runs across, and every train is one line: slope is speed, and an arrival is where a line meets a rule.

On our data the rules are levels of free space and the terminus is the threshold. Each series is a train. Promcast's nine quantile paths leave "now" as a bundle of hairlines; `predict_linear` is one ruled, dashed line. db-01's bundle reaches the page rule 45 minutes before the ruler does; logs-01 is the honest loss, where the ruler gets there first; ingest-02's ruler arrives while the bundle turns back up, a page that never needed sending.

Taken from Marey: the shared clock, true spacing, thin lines, arrivals read where a line meets a rule. Not taken: his lettering, the plate itself, or anything photographic.

## Hierarchy

1. The headline: when the first series pages, and how late `predict_linear` is ("db-01 pages at 15:15. predict_linear: 16:00, 45 min late.").
2. The `scan` command, once, directly under it. The `--torch-backend cpu` flag is a note beside it, so the command never wraps on a phone.
3. The reading over the sheet, one line with its values in ink ("db-01 · Promcast 15:15 · predict_linear 16:00 · 45 min unpaged"), and how to move the hand beside it on one line in small grey type, both in the first screen. On a phone the reading sits above the sheet. The hand's tab carries the time, so the reading does not repeat it.
4. The sheet at full width, zoomed so the 45 minutes between the two stamps is the widest mark on it.
5. Under the sheet: the arrivals table (every series, Promcast's time and the ruler's) and the key.

## The sheet

- Time runs across: one hour of measured history, then the forecast through 17:45, ten minutes past the fleet's last arrival (`AHEAD_S`; a test holds every arrival on the sheet). The social card stops at 17:00. Ten-minute columns, hour columns. On a 1440 px screen the 45 minutes between db-01's stamps is about 200 px.
- Free space runs **up** the sheet on a log scale, from 68 % at the foot to 2.2 % at the head, so the last few percent get room, a filling disk climbs, and the stamps have paper above the rule.
- The threshold is a 3 px red rule. The zone past it is hatched red.
- A train stops on the rule at the first forecast step at or under the threshold: the same step `promcast scan` counts as its breach, so a thread's end and the scan's time are one instant (`tests/test_timetable.py`).
- Every arrival is a dot on the rule: filled ink for Promcast, open blue for `predict_linear`. Only the chosen series prints stamps and a remark; every other time is in the arrivals column, so the rule never stacks labels.
- The whole forecast of every series is printed at 15 % ink to the sheet's edge, under everything, so the hours past the hand are not empty: logs-01's bundle reaches the rule at 17:35 even while the hand rests at 16:00. Full ink runs only up to the hand.
- Each series is one group: history and forecast fade together, so the background series sit at 40 % ink as a group and their stacked quantiles never print darker than one line. The chosen series is drawn last, over the rest, on a 3.5 px paper knock-out, so ingest-02's hump passes under db-01's bundle instead of through it; only it shows its `predict_linear` line. Choosing a row in the arrivals table swaps them and brings the sheet back into view.
- Train names are right-aligned where measured history ends, at now, set just above the highest point of their own line under the name and nudged apart with no leaders, on a paper halo. The % axis runs down the left on every width.

## Signature moment: two stamps on the rule

The breach is staged in the threads, on Marey's sheet, and every mark on the rule is data.

- A clock hand sweeps across the forecast and the threads draw in behind it. When db-01's bundle reaches the 5 % rule, an ink station stamp, `PROMCAST 15:15`, presses above the rule just left of the arrival, with a short leader down to the dot. From there to the sheet's right edge the rule runs 6 px thick and bleeds a little into the paper: the disk is past the line. It lands whole when the hand reaches Promcast's arrival.
- A solid red bar (60 % red) lies across the rule from the first arrival and grows behind the hand, printed through the same even `bleed` as the thick rule. Its remark, "45 min unpaged", stands over it on one line, knocked out of the hatching by a paper stroke round its letters, so no red text sits on red hatching: "45 min" is the largest numeral on the sheet (28 px at full width) and counts up as the hand crosses the gap, with "unpaged" at 14 px after it.
- Every Promcast arrival has a faint ring on the rule from the start, where its faint thread meets it, so logs-01's 17:35 reads as an arrival while the hand rests at 16:00.
- `predict_linear`'s blue thread arrives at 16:00 and its stamp, `LINEAR 16:00`, comes down late and off square in the clear paper just right of the arrival, with its own leader.
- For logs-01, where the ruler is first, the stamps swap sides and the remark reads "40 min where predict_linear paged first"; for ingest-02 the ruler's lone arrival reads "a page Promcast never sends". When the bar is too short for its remark (a phone, the card, logs-01), the remark sits over the stamps instead. Horizontal, on the clock, in minutes.
- Every stamp keeps 12 px of paper around it (`CLEAR`) and never covers the marks it labels: stamps sit above the rule, threads end on it.
- Stamps are double-ruled boxes, set a few degrees off square, with the station name small and spaced over the time. Each prints through its own filter: its own edge, specks where the paper's tooth missed the pad, pressure that fades across the face, and a faint second impression a hair out of register. A stamp prints with a short press (170 ms) when the hand reaches it.
- Only the chosen series prints stamps and a bar; the others keep their dots on the rule and their times in the arrivals column.
- The default frame rests with the hand on db-01's `predict_linear` arrival, so both stamps and the bar are on the sheet and the forecast past 16:00 shows only at 15 % ink. Dragging the hand on reveals the rest; dragging back lifts the stamps again.
- Opening, once, when the sheet is first in view: the hand runs from now to db-01's Promcast arrival, holds while the first stamp prints, then runs on to the ruler's arrival (4.6 s in all). Any input takes over at once.
- Controls: the clock is a native range input over the forecast half of the sheet and the time axis under it (drag, click, arrow keys, Home and End), with a grab cursor. The how-to line says so in 13 px grey beside the reading; on a coarse pointer it shrinks to "Drag the hand." under the axis, with no mention of keys. The hand is a dashed ink line with a tab on the time axis that carries its time. The arrivals rows are toggle buttons that bring another series forward with its stamps.
- The reading agrees with the sheet: a path that has reached the rule reads as the time it paged, as its thread ends there; before the page the last item says "Pages in 7m", after it the gap the bar draws, "30 min unpaged" growing to "45 min unpaged" (or "predict_linear 40 min early" where it paged first). `timetable.reading` writes the still frame's words and `site/site.js` repeats the same rule as the hand moves.

Engineering:

- SVG only. The ink is SVG filters: one `ink` filter per stamp (edge noise, paper tooth, a low-frequency pressure mask, a 30 % offset second impression), and `bleed`, which spreads the thick rule and the bar slightly and evenly. No WebGL, no canvas, no library. A fragment pass for ink spread does not clearly beat the filters at these sizes, so there is none.
- The filters use user-space regions, since a horizontal rule has a zero-height bounding box.
- The page ships a complete still frame from `scripts/timetable.py`: every mark the hand has passed is printed in the markup, so reduced motion, no script and the social card all show the settled state with both stamps and the bar. `site/site.js` only moves the hand, the sweep clip and the `in` class on marks.
- Every number on the sheet is also DOM text: the reading over it, the arrivals table, the range input's `aria-valuetext`, the SVG `<title>`.

## Tokens

Source of truth: `docs/design/tokens.json`.

| Token | Value | Use |
|---|---|---|
| paper | `#efece4` | page ground |
| ink | `#1a1916` | text, history, quantile threads, the Promcast stamp |
| ink-2 | `#5a564e` | secondary text; 6.2:1 on paper |
| grid | `#1a191629` | hour columns and level rules, never text |
| red | `#b02a15` | the threshold rule, its zone, the thick rule past the breach, the bar and its remark, Promcast arrival times in the log; 5.6:1 on paper |
| blue | `#1d4f91` | `predict_linear` and its stamp, nothing else; 6.9:1 on paper |

In a terminal the board keeps its own palette (`terminal` in the tokens): its bar must read on dark and light terminal themes alike. On the page the terminal section shows the CLI's own SVG export flush, as one dark slab with no window rim, its lead colour set in the page's paper instead of the terminal's high-visibility yellow-green, so the page keeps one palette. Its boards are drawn with a four-hour horizon (`BOARD_HORIZON_S` in `scripts/build_site.py`, which refuses to build if a breach falls past it), so the axis has no empty hours. The Compose demo's Grafana dashboard keeps Grafana's dark theme and the terminal palette: it is a capture of a running stack, shown as one.

## Type

- TikTok Sans (SIL Open Font License 1.1), one variable family for everything: condensed (`font-stretch: 75%`) at 800 for the headline and section heads, at 850 for the stamp times, normal width for text. Tabular figures throughout. Subset to Latin and served from `site/fonts/TikTokSans.woff2`.
- Martian Mono (SIL OFL 1.1) only for commands and the terminal board export.
- No third-party request.

## Sections

1. The sheet: masthead, the reading and how to move the hand, the sheet at full width, then the arrivals table and the key.
2. How to read it: the argument in three paragraphs, and the log of events in order, lit as the hand passes.
3. Terminal: "One board in your terminal". The heading shares its baseline with the four states, switched by keycap buttons; one board export, one caption under it. Under them, the three commands to run on your own series, each set like a station on its own rule.
4. Grafana: a small heading and the one sentence beside the Compose demo's first row, the chart and the two breach times in minutes. `scripts/grafana_frame.py` frames 45 minutes back to five minutes past Promcast's breach, waits for a fresh scrape, and clips to that row. The sentence says the demo pages at 0 B, not 5 %. A phone shows the two breach times only.
6. The footer: licences and the Marey credit.

## Numbers and where they come from

| Shown | Script |
|---|---|
| Breach times, time left, gaps (the headline's "45 min late", the stamps, the bar's "45 min"), row order, warming series | `scripts/synthetic_fleet.py` writes `docs/design/fleet-scan.json` (`promcast scan --json` of a seeded fleet, toto-2.0-22m) |
| History, nine quantile paths and `predict_linear`'s path per series | `scripts/fleet_paths.py` writes `docs/design/fleet-paths.json`; it refuses to run if the fleet no longer scans to the committed document |
| Terminal boards | `promcast.shell.board.scan_svg` from the scan document |
| Grafana times | `scripts/grafana_frame.py` writes `docs/design/grafana-first-screen.json` |

`scripts/build_site.py` reads those files and writes `site/index.html`, `site/social.html`, the board SVGs and the cards; `scripts/timetable.py` draws the sheet. No figure is typed into a template.

## Do not change

- One clock per sheet; never a per-series scale.
- Free space up the sheet on the log scale.
- A thread ends where the scan says the series breaches.
- Blue means `predict_linear` and nothing else. Red means the threshold and nothing else.
- Every mark on the rule is data: a stamp is an arrival, the bar is the gap between two arrivals in minutes, the thick rule is time past the line. No decoration goes on the rule.
- The sheet never holds the only copy of a number.
- Text meets WCAG AA, focus is visible, every control works from the keyboard, and motion respects `prefers-reduced-motion`.

## Regenerating the assets

```sh
uv run python scripts/synthetic_fleet.py docs/design/fleet-scan.json   # needs toto-2.0-22m
uv run python scripts/fleet_paths.py docs/design/fleet-paths.json      # needs toto-2.0-22m
uv run python scripts/build_site.py                                    # pages, then the PNG cards
```

The social preview is `site/social.html` at 1280 by 640, saved to `docs/design/social-preview.png` and `site/og.png`; the README uses it as its hero.

## Credits

- TikTok Sans, copyright 2024 TikTok Inc., SIL Open Font License 1.1, from the Google Fonts repository (`ofl/tiktoksans`); licence in `site/fonts/OFL-TikTok-Sans.txt`. Subset and instanced at `slnt=0` with fontTools.
- Martian Mono, SIL Open Font License 1.1; licence in `site/fonts/OFL-Martian-Mono.txt`.
- After Étienne-Jules Marey, *Graphical train timetable, Paris to Lyon* (1885). No reproduction of the plate is used.
