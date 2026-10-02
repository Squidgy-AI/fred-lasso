# Three-minute demo

One rule: **never say "hours" — say "three matches".** The upload cap and the licensing
mean we have 34 minutes, and a judge who catches an overclaim stops believing the numbers.
The honest framing is stronger anyway: *this scales by adding footage, not by changing
anything.*

| Time | Beat | What's on screen | What you say |
|---|---|---|---|
| 0:00–0:20 | **The question** | UI, question box empty | "Every club has footage nobody can ask anything useful. Here's a question a coach actually asks." Type: *show me all crosses that led to a goal.* |
| 0:20–0:45 | **Why it's hard** | the compiled-filter line under the answer | "That's three questions wearing a trench coat — find the goals, look at what came before, classify the final ball. One embedding search can't do it. Watch what Fred does instead." |
| 0:45–1:30 | **The engine** | result cards; click one, the 20s build-up plays | "The index found a candidate. We cut the twenty seconds *before* it — a clip the index never saw as a unit — and asked Cosmos a multiple-choice question about it." Click the Cosmos reasoning open. |
| 1:30–2:00 | **The receipts** | `python scripts/evaluate.py` output / Weave | "The audio anchor has never missed a goal — three of three. It over-fires at 43% precision, on purpose. Cosmos is the filter: here's what it vetoes." Name a real failure out loud. |
| 2:00–2:20 | **Act** | "Cut the reel" button → reel plays | "Every answer is already a cut list, so the reel is one click." |
| 2:20–2:40 | **Honesty + generality** | the Utrecht card with low confidence | "This one scores low — it's a supporter filming the crowd, not the pitch. The model saying 'I'm not sure' is the system working. Same code runs on the provided warehouse pack." |
| 2:40–3:00 | **Close** | README / repo | "Cosmos Reason and Embed1, VAST DataEngine and VastDB, CoreWeave GPUs, W&B inference and Weave — and Canary-1B, which was deployed on the stack but not wired into the pipeline until today. All footage CC BY. Repo's open." |

## The three lines that win it

1. *"We didn't trust the index to answer a compositional question. We used it to find candidates, then asked a constrained question about a window it had never seen."*
2. *"The anchor is deliberately over-eager. Recall is the cheap part; Cosmos buys the precision. Here's the measurement."*
3. *"Canary-1B was sitting on the stack unwired. We wired it."*

## Pre-flight, by 16:15

- [ ] `python scripts/health.py` green, screenshotted
- [ ] Event table populated **with Cosmos verdicts** (`python -m fred stats` shows classified types, not `unknown`)
- [ ] `python scripts/evaluate.py` run once, numbers written down — say the real ones
- [ ] Reel renders in under 10 seconds
- [ ] **Fallback video recorded** of the full run, plus a 60-second cut
- [ ] Repo pushed and openable in a private window
- [ ] `SUBMISSION.md` written via the challenge's `submission` skill

## If it breaks on stage

Narrate over the fallback video, don't apologise, keep to time. If only Cosmos is down,
the event table is already populated — the demo is reading rows, not calling the model
live, so it will keep working. If the UI dies, `python -m fred ask "..."` in a terminal
shows the same thing.
