# Booth Readiness Checklist — Project Pelagic

- **Initial audit:** 2026-09-27, read-only.
- **Follow-ups:** two rounds of approved fixes, carried out the same day.
- **Left untouched as instructed:** DB rows, lookalike gate, deck/poster, and the DSen2-CR branch.
- **DB after live fetches:** every live fetch inserts a detection row. `data/pelagic.db` was backed up beforehand and restored afterwards each time, so its MD5 is still `227299550e5cfde002d491af96569cb5`.
- **Leftover live outputs:** fetched rasters and masks remain in the gitignored `data/raw/live/` and `data/processed/`.

Items are ordered by how badly each would break a live demo, worst first.

| # | Item | Status | One-line fix / note |
|---|------|--------|---------------------|
| 1 | Frontend dev server starts | **READY** | `npm ci` done; Vite serves on :5173. `npm audit` reports 2 high-severity dev-dependency advisories, which weren't addressed. |
| 2 | 4 demo scenes render on the map | **READY** | Verified in the browser: no console or page errors, and no NaN paths. |
| 3 | Fallback screenshots match the current UI | **READY** | All 15 recaptured: demo 5/5, live 10/10. Nothing needed deleting. |
| 4 | Live fetch (CDSE + GFW + land mask) | **READY** | 3 regions × OK: Singapore, Stockholm, Chonos. Each fetch takes about 60 s, so have narration ready. |
| 5 | Backend + cached demo scenes (API) | **READY** | None needed. |
| 6 | Doc fixes reach origin | **NEEDS FIX** (one click) | Open and merge the PR: https://github.com/Zenzencat/Project-Pelagic/pull/new/docs/fix-lookalike-fp-range |
| 7 | Stale numbers: README / status.md / AGENTS.md | **FIXED on branch** | Lands on `main` when the PR in #6 merges. |
| 8 | Stale numbers: deck / poster | **NEEDS FIX** | Deck slide 6 says "61 tests" (now 81). The scene-membership wording on deck slide 12 and poster §6 disagrees with the holdout eval. |
| 9 | MANIFEST.md accuracy | **FIXED on branch** | Round 18 note committed; the `06` row now says "mostly excluded, with small slivers remaining at the island edges". |
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

### 3. Fallback screenshots (all recaptured 2026-09-27)

| Set | Source | Result |
|---|---|---|
| `demo_fallback/` ×5 | Scratchpad script with the same tile-settle wait as the repo scripts | Current areas and the honest 0-vessel state. The old mock AIS vessel names are gone. |
| `live_fallback/01–04` | `scripts/capture_live_screenshots.js` | Singapore Strait, OK. Land mask removed 61.3% of raw oil px; 5 real GFW vessels. |
| `live_fallback/05–06` | `scripts/capture_live_variety_screenshots.js` | Singapore Strait, OK (same product, same result). |
| `live_fallback/07–08` | same | Stockholm Archipelago, OK. `S1D_…20260812T162035…`; land mask removed 35.1% (128,315 px via OSM island refinement); 5 real vessels. |
| `live_fallback/09` | same | Chonos Archipelago, OK. Land mask removed 62.6%; GFW honestly reports no AIS presence within 10 km. |
| `live_fallback/10` | Labeled crop of `08`, built from marker positions recorded during that same run | 5 distinct vessel markers. The caption count is taken from the run, not copied from the old image. |

- Playwright 1.62.1 was run from the existing npx cache via `NODE_PATH`; no new install.
- Pre-recapture copies of both folders are saved in the session scratchpad.
- **Booth Q&A note:** Singapore still shows some slick polygons on reclaimed land (Pasir Panjang terminal, Jurong). The OSM land mask doesn't cover all recent reclamation, so expect the question.

### 6–7. Git state

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
