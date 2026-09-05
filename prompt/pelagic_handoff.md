# Project Pelagic — Handoff: Supplementary Evidence Integration, Post-QC

## Status

Your supplementary evidence work (ERA5 wind, multi-temporal SAR, Sentinel-2 optical,
plus the DSen2-CR scoping decision) went through a full QC pass and a follow-up fix
round. Both are merged/ready. Quick summary so you have the full picture, not just
the next task:

**QC pass (against `0ae4506`)** — all 4 tracks matched the handoff brief. No mocked
data standing in for real, no oversold claims, detection-only scope respected,
dB-aware SAR preprocessing untouched. 41 tests passing. Found 8 CONCERN items, 0 FAIL.

**Fix round (`fix/qc-followups`, 5 commits, ready to push/merge)** — closed 4 of the
8 concerns:
- Land-mask vs. polygon coordinate mismatch on non-square-pixel bboxes (was up to
  11.1km offset on some scenes — now 0m, verified against an independent oracle)
- Comparison-candidate dedup was counting the same acquisition twice as "2
  candidates" — now correctly dedupes by acquisition time
- Stale docs (AGENTS.md, predict_empty_masks.md, the Kaggle kernel's preprocessing
  copy) brought back in sync with what the code actually does now
- One unguarded DB write that could 500 a successful detection on a serialization
  failure — now degrades gracefully and logs instead of failing

51 tests passing total, byte-identical regression against the pre-fix baseline on
real holdout scenes with the real v2 checkpoint. Nothing here touches your work
directly — it's all in the integration layer around it.

## What's left

**1. Credential verification (blocks live-fetch reliability — not part of this ticket)**
The one thing nobody can verify without live CDSE credentials: whether Sentinel
Hub's real Process API response names products the way `cdse_fetch.py` expects
(`_COG` suffix vs. `sentinel1ProductId`). If it doesn't match, every live fetch
fails. There's already a prompt for this at `prompt/pelagic-credential-verification.md`
— whoever has live credentials on hand should run it before we plan a live demo
around fetch. Flagging it here so you know it's outstanding, but it's not the task
below.

**2. Preview storage: move from DB to disk — this is your next task**
Preview images (SAR/optical PNGs) are currently stored as base64 straight inside
`data/pelagic.db`, which is a tracked git file. At ~2-5MB per live detection, that
DB will keep growing every time someone runs a live fetch, and it breaks the
project's own convention — we already gitignore `data/external/`, `data/raw/live/`,
and old diagnostic arrays specifically to keep large binary data out of git. Moving
previews to disk with just a path reference in `supplementary_json` fixes it
cleanly, with no loss of functionality (the offline demo safety net is a separate
screenshot set, not these DB blobs).

Prompt below is ready to run as-is — self-contained, scoped, doesn't touch the
credential item.

---

### Prompt to run

```
PROJECT: Pelagic — fix QC-flagged issue: preview storage location (item 7)

CONTEXT
QC pass on commit 0ae4506 found that supplementary_json stores PNG
preview images as base64 data URIs directly in data/pelagic.db (a
tracked SQLite file), at ~530KB for a grayscale SAR preview and up to
~1.58MB for a BGR overlay — roughly 2-5MB per live detection once
comparison/optical previews are included. This breaks the project's
own established convention: data/external/, data/raw/live/, and the
phase3_6 .npz diagnostics are all gitignored specifically to keep
large/regenerable binary data out of git. Decision made: move preview
storage out of the DB entirely, onto disk, with only a path/reference
stored in supplementary_json.

TASK 1 — Locate the current implementation
- Find every place that builds a base64 preview and writes it into
  supplementary_json (live_fetch path at minimum — check eval/scripts
  too in case they share the same encoding function)
- Find how the frontend currently consumes these previews (likely an
  <img> tag reading a data: URI directly out of the API response) —
  report the exact component/file before changing anything

TASK 2 — Move previews to disk
- Write preview PNGs to data/raw/live/previews/<detection_id>_<kind>.png
  (create the directory if needed; confirm it falls under the existing
  data/raw/live/ gitignore entry, add a specific entry if it doesn't)
- Replace the base64 data URI in supplementary_json with a reference
  the frontend can resolve (a relative path or an API-served URL —
  pick whichever matches how this project already serves any other
  static/generated assets, and say which convention you followed)
- Add a serving route if one doesn't already exist, so the frontend
  can actually load the image from the new location

TASK 3 — Update the frontend
- Point whatever currently reads the inline data URI at the new
  reference instead
- Confirm previews still render (a scripted check via
  scripts/verify_supplementary_ui.py or equivalent counts, since this
  project doesn't have a human clicking through the UI in this pass)

TASK 4 — Backward compatibility
- Existing rows in data/pelagic.db already have inline base64 previews
  from before this fix — do NOT migrate or rewrite them. New writes go
  to disk; old rows keep working as-is until they age out naturally.
  Report the current DB size before/after so Zen can decide separately
  whether a one-time historical cleanup is worth doing.

TASK 5 — Cleanup/cap
- data/raw/live/previews/ will grow unboundedly with usage, same as
  data/raw/live/ scene cache already does. Add a simple cap (e.g. max
  total size or max file count, evicting oldest first) — keep it
  configurable via a constant near the top of the relevant module, not
  hardcoded inline. Pick the simplest approach that fits the existing
  cache/cleanup pattern in this codebase if one already exists;
  otherwise propose one and justify it briefly.

VERIFICATION
- Re-run the full test suite — must still pass
- Re-run the same end-to-end /api/predict checks used in the last two
  QC rounds (oil_00000, no_oil_00004, lookalike_00000, filter True/False)
  and confirm: previews are now written to disk, supplementary_json
  contains a reference (not base64), and the response still lets the
  frontend display a preview
- Report exact before/after size of a single live detection's
  supplementary_json payload, and current data/pelagic.db size

Do not touch item 1 (cdse_fetch product-name matching) — that still
needs the credential verification pass and is out of scope here.
```

---
