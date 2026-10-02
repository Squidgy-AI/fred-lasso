# Fred Lasso

**A football video analyst agent.** Ask a compositional question about hours of football
and get back clips, with the reasoning that justified each one.

> *"Show me all the crosses that led to a goal."*

That question is the whole point. A single embedding search cannot answer it, because it
is three questions wearing a trench coat: **find the goals**, **look at what came
before**, **classify the final ball**. Fred decomposes it.

Built for the **VAST Builders Challenge — Real-Time Video Agents Hack**.

---

## How it works

```
TIER 1  recall                       the pre-built challenge stack
  video -> 5s segments -> YOLO11 + Cosmos3-Reason captions + Cosmos Embed1 vectors -> VastDB
                                              |
TIER 2  precision                             |  anchors (cheap, layered, independent)
  crowd-audio energy spike  -----------------+   sustained roar above a rolling baseline
  Canary-1B ASR confirmation ----------------+   "goal", "scores" in the commentary
                                              |
                                              v
  stitch the 20s BEFORE each anchor into one clip the index never saw as a unit
                                              |
                                              v
  NVIDIA Cosmos3-Reason, CONSTRAINED multiple choice:
     Q1  was a goal actually scored here?            (verifier - kills false anchors)
     Q2  what was the final ball?                    (classifier - fixed 6-way taxonomy)
                                              |
                                              v
  EVENT TABLE  (VastDB, mirrored to SQLite)
  match | t_anchor | goal_confirmed | assist_type | confidence | anchor_sources | clip_path
                                              |
             +--------------------------------+------------------+
             v                                                   v
  AGENT: question -> filter (W&B serverless inference)    ACT: highlights reel / alert
  semantic fallback to hybrid search for fuzzy questions
```

**The design decision worth defending:** the LLM never answers the football question. It
*compiles* the question into a filter, which is then executed deterministically against
rows that Cosmos verified. That keeps every answer grounded in evidence and makes the
agent explainable — the UI shows you the filter it produced.

**The second one:** the audio anchor is deliberately over-eager. It is tuned for recall,
and Cosmos is the precision filter. Measuring that trade-off is the evaluation.

---

## Quickstart (on the challenge VM)

```bash
git clone <this repo> && cd vast-football-agent
pip install -r requirements.txt          # ffmpeg + ffprobe must already be on PATH

python scripts/health.py                 # is every endpoint answering?
python scripts/fetch_footage.py          # CC-licensed football from Wikimedia Commons
python scripts/build_match.py            # stitch the U-17 clips into one match
python -m fred build data/raw/*.mp4 data/raw/*.webm --vastdb
python -m fred ask "show me all crosses that led to a goal"
python -m fred serve                     # the demo UI on http://127.0.0.1:8800
```

Credentials are read exactly the way the challenge Cursor skills read them: the single
`/config/<team>.config` file, environment first. Nothing is ever printed or committed.
Off-VM, every sponsor call degrades gracefully instead of crashing, so you can develop
the whole pipeline locally with `--no-cosmos`.

---

## Inputs

| Input | Where | Notes |
|---|---|---|
| Source video | `data/raw/*.{mp4,webm}` | any length; audio track required for anchors |
| Ground truth | `data/labels/ground_truth.csv` | hand-labelled, `scripts/label.py` writes it |
| Credentials | `/config/<team>.config` | provided on the VM; never committed |
| Taxonomy | `fred/config.py: ASSIST_TYPES` | the closed option set Cosmos must choose from |
| Tier-1 prompt | `fred/vss.py: FOOTBALL_INGEST_PROMPT` | 623 chars; names every event we later retrieve |

## Outputs

| Output | Where |
|---|---|
| Event table | `data/out/fred_events.sqlite`, VastDB `fred-events`, `data/out/events.jsonl` |
| Build-up clips | `data/clips/<match-slug>_<seconds>.mp4` |
| Highlights reel | `data/out/highlights.mp4` |
| Eval report | `data/out/eval_report.json` + a Weave evaluation |
| Attribution | `data/out/footage_manifest.json` |

**Event row:** `event_id, match, source_video, event_type, t_anchor, window_start,
window_end, goal_confirmed, assist_type, confidence, anchor_sources, anchor_score,
cosmos_letter, cosmos_reasoning, cosmos_model, clip_path, clip_s3_uri, asr_text`.

---

## Evaluation

The claim is that decomposition beats one embedding search. That is only worth saying if
it is measured, so the same hand-labelled set runs through progressively more of the
system:

| Stage | What it is | What it isolates |
|---|---|---|
| **A** | anchors only — every crowd spike called a goal | recall of the audio signal |
| **B** | anchors + Cosmos goal gate | how much precision the verifier buys |
| **C** | B + constrained multiple choice | accuracy on the actual question |
| **D** | the question straight into hybrid search | the baseline we claim to beat |

```bash
python scripts/label.py                      # hand-label; a model must not label its own eval
python scripts/evaluate.py --with-semantic   # runs A-D, logs to Weave
```

Metrics: precision / recall / F1 for goal detection, accuracy plus a confusion matrix for
assist type. Per-anchor predictions and the aggregate land in Weave so the stages sit
side by side and a judge can click into any clip.

**Measured so far** (3 matches, 7 labelled anchors, 3 real goals, Cosmos not yet run):

```
stage                              precision  recall     f1   tp/fp/fn
A_anchors_only                         0.429   1.000  0.600   3/4/0
```

The audio anchor has not missed a goal yet — 3 of 3, across 34 minutes of footage — and
over-fires at 43% precision. Every false positive so far is pre-match footage filmed
outside the ground, which is precisely the kind of "loud but not football" moment stage B
exists to veto. Ground truth is independently verifiable: two of the three goals carry
the uploader's own burned-in caption ("1-0 van de Streek", "1-2 Venema"), recorded in the
`evidence` column of `data/labels/ground_truth.csv`.

---

## Footage and licensing

Every football video with a scoreboard burned into it is a broadcast, and every broadcast
is somebody's exclusive right. So this project uses none.

| Source | Licence | Use |
|---|---|---|
| 2018 FIFA U-17 Women's World Cup, NZ v Canada — 9 clips, 4 min | **CC BY-SA 4.0** | pitch action, assist classification |
| FC Utrecht v Ajax, KNVB Cup SF Mar 2020 (2-0), 29 min — [FCUFAN](https://www.youtube.com/@fcufan) | **CC BY 3.0** | anchor detection over a long video |
| FC Utrecht v Ajax, Eredivisie Dec 2018 (1-3), 5 min 4K — FCUFAN | **CC BY 3.0** | second anchor test, held-out goal |

Both via [Wikimedia Commons](https://commons.wikimedia.org/wiki/Category:Videos_of_association_football_matches).
Attribution is recorded in `data/out/footage_manifest.json` and shown in the UI footer.

**Deliberately not used:** SoccerNet (its terms prohibit "public screening" and
"publication of source videos, clips, frames"), the DFL Bundesliga Shootout set (rights
unclear; derivative re-uploads carry licence tags their uploaders cannot grant), and
archive.org match rips. StatsBomb Open Data is safe and is the right source for the
taxonomy, but ships no video.

---

## Sponsor stack

| Tool | What it actually does here |
|---|---|
| **NVIDIA Cosmos3-Reason** | the constrained second pass: verifies the goal, classifies the final ball |
| **NVIDIA Cosmos Embed1** | 256-d hybrid text+visual search for fuzzy questions |
| **NVIDIA Canary-1B** | commentary ASR to confirm anchors — *deployed on the stack but not wired into the pipeline, so we wired it* |
| **VAST** | DataEngine pipeline, S3, and VastDB for our own event table alongside the vectors |
| **CoreWeave** | the GPUs all of the above run on |
| **W&B inference + Weave** | question→filter compilation; tracing and the ablation evaluation |
| **Cursor** | how the day was built |

---

## Layout

```
fred/
  config.py     credentials and the taxonomy          cosmos.py   the constrained second pass
  anchors.py    crowd audio + Canary ASR              clips.py    ffmpeg: cut, stitch, split
  events.py     event table (VastDB + SQLite)         agent.py    question -> filter -> answer
  pipeline.py   the end-to-end loop                   evaluate.py the ablations
  vss.py        the pre-built search/ingest API       trace.py    optional Weave tracing
scripts/        health, fetch_footage, build_match, label, evaluate
app/            the demo UI (no build step, no dependencies)
docs/PLAN.md    the plan this was built from, including the risks
```

## Known limits

- Anchors need crowd noise. Silent or empty-ground footage falls back to Tier-1 captions.
- A 5-second segmentation means a 20-second window is four segments; very long build-ups
  get truncated.
- Cosmos latency is tens of seconds per clip on a shared cluster, so the demo plays
  pre-computed rows. Anchor detection itself is real time (29 min of audio in 6.6 s).
- `assist_type` is only as good as the camera. Wide tactical shots classify well;
  crowd-facing fan footage does not, and low confidence is the model saying so.
