# Booth Readiness Checklist — Project Pelagic

- **Initial audit:** 2026-09-27, read-only.
- **Follow-ups:** two rounds of approved fixes, carried out the same day.
- **Third round, same day: SAR channel-order fix** (branch `fix/live-channel-order`).
  - Live scenes had been reaching the U-Net with VV/VH swapped relative to
    training. Full write-up: `docs/CHANNEL_ORDER_INVESTIGATION.md`.
  - The live fetch now writes `[VH, VV]`.
  - All 10 live fallback screenshots were recaptured on the fixed pipeline.
  - Evaluator figures were relabelled and regenerated.
  - Detection #43 is marked as pre-fix output.
- **Left untouched as instructed:** lookalike gate, deck/poster (including
  the `deck_assets/` copies of the evaluator figures and `MANIFEST.md`), and
  the DSen2-CR branch.
- **DB:**
  - Live fetches were rolled back again: backed up beforehand, restored to MD5
    `227299550e5cfde002d491af96569cb5` afterwards.
  - The only intended change is then applied: #43's `supplementary_json` gets
    the channel-order note.
  - New MD5 `e29370a00eca6a8b7f1c52ea8a28ce60`, still 7 detection rows.
  - The file grew ~370 KB. That's freed SQLite pages from rewriting #43's
    375 KB JSON row (content +618 bytes); no VACUUM was run.
- **Leftover live outputs:** fetched rasters and masks remain in the gitignored
  `data/raw/live/` and `data/processed/`.
  - Files in the **main checkout** are pre-fix `[VV, VH]`; see
    CHANNEL_ORDER_INVESTIGATION.md §4a.
  - The one post-fix raster is in this worktree.

Items are ordered by how badly each would break a live demo, worst first.

| # | Item | Status | One-line fix / note |
|---|------|--------|---------------------|
| 1 | Frontend dev server starts | **READY** | `npm ci` done; Vite serves on :5173. `npm audit` reports 2 high-severity dev-dependency advisories, which weren't addressed. |
| 2 | 4 demo scenes render on the map | **READY** | Verified in the browser: no console or page errors, and no NaN paths. |
| 3 | Fallback screenshots match the current UI | **READY** | Live 10/10 recaptured again on the channel-order-fixed pipeline. Demo 5/5 are unchanged, since cached holdout scenes were never affected. |
| 4 | Live fetch (CDSE + GFW + land mask) | **READY** (after the PR in #6 merges) | 3 regions × OK on the fixed pipeline: Singapore, Stockholm, Chonos. Land mask and GFW both verified. Slicks are now much smaller (Singapore 1.3 km², was 22.3), see Q&A note in §3. |
| 5 | Backend + cached demo scenes (API) | **READY** | None needed. |
| 6 | Channel-order fix reaches origin | **NEEDS FIX** (one click) | Open and merge the PR for branch `fix/live-channel-order`: https://github.com/Zenzencat/Project-Pelagic/pull/new/fix/live-channel-order. The previous `docs/fix-lookalike-fp-range` PR is merged (#10). |
| 7 | Stale numbers: README / status.md / AGENTS.md | **FIXED on branch** | Channel order, open items and the `no_oil_00005` cause are recorded. Lands on `main` with #6. |
| 8 | Stale numbers: deck / poster | **NEEDS FIX** | Deck slide 6 says "61 tests" (now 82). The scene-membership wording on deck slide 12 and poster §6 disagrees with the holdout eval. **New:** the deck's copies of the `holdout_v*_viz_*` figures still have the near-black "SAR VV Channel" panel (actually VH); corrected versions are in `docs/`. Any deck claim about live-fetch slick sizes predates the fix. |
| 9 | MANIFEST.md accuracy | **NEEDS FIX** (deck asset, left untouched) | The `live_fallback` rows (lines 67–85) describe pre-fix results (e.g. "781,193 of 2,641,623 raw px (29.6%)", `S1C…20260813` for Stockholm, `MARINA GENESIS`… vessel names). They no longer match the recaptured images; see §3 for the current values. |
| 10 | Lookalike gate threshold 0.975 | **READY** (with a caveat) | Hardcoded at `src/analysis/lookalike_filter.py:42`, not a saved artifact. Describe it that way at the booth. |
| 11 | Teammate contribution QC | **NEEDS FIX** (process) | No QC exists for PR #8 (milm01) or the DSen2-CR branch. Your own commit `a2f8c40` (ERA5 re-confirm) isn't covered by any QC report either. |
| 12 | Legacy mock rows 37/38 in DB | **NEEDS FIX** (low risk) | Unreachable from the sidebar. Left untouched as instructed. |

---

## Details

### 1–2. Frontend and the 4 demo scenes
In-browser check, clicking each card in turn. Overlay paths are the polygon fragments plus 1 footprint.

| Scene | Overlay paths | Area badge | Vessels | Rendering errors |
|---|---|---|---|---|
| oil_00000 | 38 | 159.8 km² | 0 | none |
| oil_00001 | 28 | 13.5 km² | 0 | none |
| oil_00003 | 10 | 48.1 km² | 0 | none |
| oil_00004 | 233 | 126.1 km² | 0 | none |

- `frontend/dist/` is still a stale 08-24 build. Use `npm run dev`, not `dist/`.

### 3. Fallback screenshots

**Live 01–10 recaptured again (2026-09-27, third round), after the channel-order fix.**

| Set | Source | Result |
|---|---|---|
| `live_fallback/01–04` | `scripts/capture_live_screenshots.js` | Singapore Strait, OK, `S1D_…20260810T112444…`. Land mask removed 1,162 of 14,327 raw oil px (8.1%). 5 real GFW vessels: JMS BENAR, VB MENANG, OKEE JOHN T, PILOT GP57, NOBLE VEGA. Slick area 1.3 km². |
| `live_fallback/05–06` | `scripts/capture_live_variety_screenshots.js` | Singapore Strait, OK (same product, same result). |
| `live_fallback/07–08` | same | Stockholm Archipelago, OK, `S1D_…20260812T162035…`. Land mask removed 11,367 of 41,272 raw px (27.5%; 453 px via OSM island refinement). 5 real vessels. |
| `live_fallback/09` | same | Chonos Archipelago, OK, `S1D_…20260814T234857…`. Land mask removed 21,247 of 57,628 raw px (36.9%). GFW honestly reports no AIS presence within 10 km. |
| `live_fallback/10` | Labelled crop of `08` built from the 5 marker positions recorded during that same run | 5 distinct vessel markers. The caption count comes from the run. |

- **How it was run:**
  - Repo scripts run as scratchpad copies with two read-only additions: saving
    each `/api/live/fetch` response and recording marker positions for `10`.
    Screenshot steps are identical.
  - Playwright 1.62.1 from the existing npx cache via `NODE_PATH`; no new
    install.
  - Backend: this branch's code on :8000. Frontend: the main checkout's
    Vite server on :5173; `frontend/src` is identical to the branch apart
    from line endings.
  - `.claude/launch.json` was temporarily extended and then reverted.
- Pre-recapture copies of `live_fallback/` are saved in the session scratchpad.
- `demo_fallback/` ×5 wasn't recaptured this round. Cached holdout scenes read
  Part III files, which already use the training layout, so they're unaffected.
- **Booth Q&A notes:**
  - *"Why are the live slicks so much smaller than in older material?"*
    Before 2026-09-27 the live pipeline fed the model VV/VH swapped. The same
    Singapore product went from 14.8% of the scene flagged to 0.34%.
    Older live numbers (deck, MANIFEST, Detection #43) are pre-fix.
  - The remaining Singapore fragments mostly hug island shorelines. The earlier
    reclaimed-land polygons (Pasir Panjang, Jurong) no longer appear.

### 6–7. Git state

**Current (third round):**
- `fix/live-channel-order` is **pushed**, branched from `origin/main` at
  `e7bba98` (PR #10 merged).
- **To open the PR:**
  - Link: https://github.com/Zenzencat/Project-Pelagic/pull/new/fix/live-channel-order
  - Or: `gh pr create --base main --head fix/live-channel-order`
- Tests on this branch: **80 passed**. The 2 remaining collected tests are
  the Windows-only symlink-privilege failures in `test_preview_storage.py`;
  `-k "not symlink"` gives 80 passed.
- **`no_oil_00005` (read-only finding, not fixed):** its 100% false positive
  comes from the dB/linear test in `preprocess_for_prediction()` routing an
  all-zero scene to the linear branch. The committed JSON came from
  `compare_checkpoints.py`'s always-dB path. Details are in status.md
  "Evaluation".

**Previous rounds (historical):**

| Branch | Where | Commits beyond `832d858` |
|---|---|---|
| `main` (local) | 2 ahead of origin; the push was rejected by a repository rule, which I didn't try to bypass | `a2f8c40` ERA5 re-confirm, `aa07f23` README/status.md 1.9–44.7% + v3 `.gitignore` |
| `docs/fix-lookalike-fp-range` | **pushed** | the above + `f437c08`: all 15 recaptured screenshots + AGENTS.md "Known trap" fix |
| `experiment/v3` | **pushed** | `a2f8c40`, `aa07f23` + `1a2d842`: v3 kernels, joint U-Net, comparison scripts, results. No v3 checkpoints (gitignored). |

**To open the PR:**
- Link: https://github.com/Zenzencat/Project-Pelagic/pull/new/docs/fix-lookalike-fp-range
- Or with the GitHub CLI: `gh pr create --base main --head docs/fix-lookalike-fp-range --title "docs: correct lookalike FP range; recapture fallback screenshots"`

**After merging:** run `git switch main && git pull`. Local `main` is a strict ancestor of the branch, so this fast-forwards.

- The branch also carries a follow-up commit: the MANIFEST.md Round 18 note, the corrected `06` row, and this file.

**`a2f8c40` (ERA5 re-confirm, 2026-09-14):** 4 files.
- `data/pelagic.db`: adds live Detection #43.
- `docs/live_era5_verification_2026-09-14.json`: new artifact. It contains no credential values; "key" appears only in prose.
- `docs/status_history.md`: +53 lines.
- `docs/ai_technique.md`: +14 lines.
- Its only QC report is `prompt/pelagic_handoff.md`, dated 2026-09-05 and scoped to `0ae4506`, so it predates this commit. Its only self-verification is recorded in its own commit message (tests/test_era5.py 8 passed; one live fetch returning ERA5 `available`, 4.76 m/s).

**Still untracked on `main`:** the deck, `deck_assets/`, `output/poster/`, and the phase-4 shape-filter work, which is gate-adjacent and deliberately left out of v3

### 8. Remaining stale numbers (deck/poster excluded by instruction)
| Location | Stated | Current |
|---|---|---|
| Deck slide 6 | 61 tests | 81 tests collected |
| Deck slide 12 + poster §6 | 7 correctly suppressed scenes, "all on land" | The count matches, but the scene membership differs between the holdout eval set {00002, 00004, 00005, 00006–09} and MANIFEST's live-app set {00002, 00003, 00004, 00006–09} |

### 10–12. Unchanged from the initial audit
- **Gate:** `GATE_THRESHOLD = 0.975` is hardcoded and opt-in (`apply_lookalike_filter=False`). `lookalike_classifier_final.joblib` loads fine in the local venv (sklearn 1.9.0).
- **QC:** a QC report exists for papajittan's PR #4 (`prompt/pelagic_handoff.md`: 8 CONCERN, 0 FAIL, all since addressed). There is none for milm01's PR #8 or the unmerged `origin/experiment/dsen2cr-kaggle-feasibility`.
- **Rows 37/38:** legacy mock rows with a shallower GeoJSON format that crashes `getFitBounds()` if selected. The sidebar never shows them.
- **ERA5 / Sentinel-2 / temporal evidence:** not exercised in this audit. Run one fetch with them ticked before the booth if you plan to show them.
