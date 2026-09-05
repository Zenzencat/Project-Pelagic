# PROJECT PELAGIC — CODEX LOOP ENGINEERING AGENT

You are working autonomously inside the existing **Project Pelagic** repository.

Your job is NOT to make one large speculative implementation.

Instead, operate using a strict engineering loop:

**INSPECT → PLAN → IMPLEMENT SMALL CHANGE → TEST → VERIFY REALITY → REVIEW → LOOP**

Continue looping until the current task satisfies its acceptance criteria, encounters a genuine external blocker, or reaches an explicit stop condition.

Never fabricate successful results, API responses, timestamps, satellite scenes, credentials, measurements, tests, or integrations.

---

# 0. CORE OPERATING RULE

For every task, repeatedly execute this loop:

## LOOP

### 1. INSPECT
Before changing code:

- inspect the repository structure
- inspect relevant existing files
- inspect existing tests
- inspect `.env.example`
- inspect existing authentication/fetch utilities
- inspect existing API response models
- inspect frontend components consuming those responses
- inspect `docs/status.md`
- inspect git diff/status if available

Do not assume the prompt's filenames or architecture are perfectly current.

The repository is the source of truth.

### 2. DEFINE CURRENT GAP

State internally:

- what currently exists
- what is missing
- what can be reused
- what assumptions need verification
- what the smallest useful next change is

Avoid rewriting working systems.

Prefer extending existing abstractions.

### 3. PLAN ONE SMALL ITERATION

Choose a bounded change that can be independently tested.

Good:

- add ERA5 client wrapper
- add one response field
- reuse one Sentinel fetch helper
- add one catalog search function
- add one UI panel

Bad:

- rewrite the entire live-fetch pipeline
- introduce a new architecture without evidence it is needed
- implement all four tracks simultaneously

### 4. IMPLEMENT

Make the smallest coherent code change.

Follow existing project conventions.

Reuse existing code wherever possible.

Do not duplicate:

- CDSE authentication
- Sentinel Hub request logic
- environment loading
- inference pipeline
- API serialization

unless repository inspection proves reuse is impossible.

### 5. TEST

After every meaningful change, run the narrowest relevant validation first:

- syntax/type checks
- unit tests
- target module tests
- API tests
- integration tests

Then run broader tests if appropriate.

Do not continue stacking code on top of a failing change.

### 6. VERIFY REALITY

For anything involving external data, distinguish:

- code works
- credentials work
- remote API works
- real data was returned
- returned data corresponds to the requested real scene

These are separate claims.

Never turn:

"the code path should work"

into:

"the integration works."

If credentials or network access are unavailable, mark the external verification as **SKIPPED/BLOCKED**, not PASS.

### 7. SELF-REVIEW

Before proceeding, inspect your own diff and ask:

- Did I duplicate existing code?
- Did I accidentally fabricate a fallback value?
- Could errors be mistaken for valid scientific data?
- Did I preserve real acquisition timestamps?
- Did I expose uncertainty honestly?
- Did I add unnecessary complexity?
- Could this break the existing live-fetch pipeline?
- Are API/UI labels scientifically honest?
- Are tests validating behavior rather than implementation details?

Fix issues before continuing.

### 8. LOOP OR EXIT

If acceptance criteria are not satisfied:

return to **INSPECT** using what was learned.

If they are satisfied:

complete the task's documentation and proceed to the next priority task.

---

# 1. GLOBAL SAFETY / SCIENTIFIC-INTEGRITY RULES

This system deals with real Earth-observation data.

Therefore:

## NEVER FABRICATE DATA

Never fabricate:

- acquisition timestamps
- latitude/longitude
- Sentinel product IDs
- ERA5 wind values
- satellite images
- cloud percentages
- comparison passes
- inference outputs
- credentials
- API success

A missing value is better than a fake value.

Use explicit statuses such as:

- `available`
- `unavailable`
- `skipped`
- `blocked`
- `not_configured`
- `no_match`

when appropriate.

## DO NOT SILENTLY CLASSIFY OIL

The supplementary analyses are evidence for a human reviewer.

Do NOT silently turn:

- ERA5 wind
- temporal persistence
- Sentinel-2 appearance

into labels such as:

- confirmed oil
- likely oil
- false positive
- lookalike

unless an existing explicitly designed classifier already does so.

Surface evidence.

Let the reviewer interpret it.

## PRESERVE EXISTING WORKING BEHAVIOR

The live Sentinel-1 inference path is already useful.

New features should degrade gracefully.

If an optional supplementary source fails, the primary detection should still work whenever possible.

---

# 2. REPOSITORY DISCOVERY — DO THIS FIRST

Before Task 1, inspect enough of the repository to understand:

1. live-fetch endpoint implementation
2. CDSE authentication
3. Sentinel Hub Process API usage
4. inference entry point
5. detection result schema
6. frontend displaying live detections
7. configuration/environment management
8. testing framework
9. existing analysis scaffolds:
   - `src/analysis/era5_wind_check.py`
   - `src/analysis/sentinel1_revisit_check.py`
10. `docs/status.md`

Also inspect `.env.example`.

Do not invent environment variable names before checking it.

Produce an internal dependency map showing approximately:

`live fetch → Sentinel scene → inference → detection result → API → UI`

and identify where optional supplementary analysis should plug in.

Then begin Task 1.

---

# TASK 1 — ERA5 WIND CROSS-CHECK

Priority: **HIGH / FIRST**

## OBJECTIVE

For a real live-fetched Sentinel-1 detection with:

- real acquisition datetime
- real geographic location / detection centroid

retrieve ERA5 10 m wind components for approximately that time/location and expose the resulting wind speed.

Use:

`speed = sqrt(u10² + v10²)`

Do not invent values when ERA5 is unavailable.

---

## TASK 1 LOOP

Repeat the engineering loop until the following acceptance criteria are met or a genuine blocker is proven.

### Phase A — Understand Existing Scaffold

Inspect:

`src/analysis/era5_wind_check.py`

Determine:

- what is already implemented
- what is stubbed
- whether its assumptions still match the repository
- whether dependencies already exist
- where live-fetch results expose acquisition time and coordinates

Do not rewrite working scaffold code merely for style.

### Phase B — Configuration

Inspect `.env.example`.

Use existing CDS environment variable conventions if present, especially:

- `CDSAPI_URL`
- `CDSAPI_KEY`

Do not silently substitute fake defaults.

If configuration is missing:

return a clear skipped/not-configured result.

### Phase C — ERA5 Retrieval

Implement the smallest reusable interface that:

1. receives acquisition datetime + location
2. requests ERA5 hourly single-level data
3. retrieves:
   - 10m u-component
   - 10m v-component
4. extracts the relevant spatial/time point
5. computes wind speed in m/s
6. returns data plus provenance/status information

Keep ERA5's coarse spatial resolution explicit.

### Phase D — Integration

Wire ERA5 analysis into the live-fetch path as an OPTIONAL supplementary step.

Primary Sentinel inference must not become dependent on successful ERA5 retrieval.

Expose something equivalent to:

`ERA5 wind speed at acquisition: X m/s`

when available.

If unavailable, show the reason instead of `0`, `None` masquerading as data, or fabricated values.

### Phase E — Verification

Test:

1. pure wind-speed calculation
2. missing credentials
3. malformed/failed ERA5 response
4. successful mocked API result if appropriate
5. live endpoint integration
6. UI/API serialization

If real CDS credentials and network access are available:

perform one real request against a real live-fetched detection.

Verify:

- requested timestamp comes from the real Sentinel scene
- requested coordinates match the detection/location
- returned ERA5 record has the expected time
- wind magnitude is physically plausible

A weather-history comparison may be used as a loose sanity check, not as proof of exact equivalence.

If real credentials are unavailable:

mark end-to-end external verification **SKIPPED**, while still testing the code path locally.

### TASK 1 ACCEPTANCE CRITERIA

Task 1 is complete only when:

- existing live inference still works
- ERA5 integration is optional
- no fake wind value can appear
- wind speed uses real u/v data
- missing credentials fail honestly
- API exposes the result/status
- UI displays the result/status
- relevant tests pass
- `docs/status.md` is updated
- limitations mention ERA5's coarse resolution relative to Sentinel-1 imagery

Then inspect the final diff before moving to Task 2.

---

# TASK 2 — MULTI-TEMPORAL SENTINEL-1 COMPARISON

Priority: **HIGH**

## OBJECTIVE

Given a real live Sentinel-1 detection:

- search for Sentinel-1 GRD observations covering the same area
- find useful passes on different dates
- retrieve one or more valid comparisons
- run the normal inference pipeline on them
- present the results side-by-side

Do NOT automatically label the patch persistent/transient or real/fake oil.

---

## TASK 2 LOOP

### Phase A — Inspect Existing Code

Inspect:

`src/analysis/sentinel1_revisit_check.py`

and the current live Sentinel fetch implementation.

Identify reusable functions for:

- authentication
- catalog querying
- Process API fetching
- image preprocessing
- inference
- result serialization

Reuse these.

Do not build a second parallel Sentinel pipeline.

### Phase B — Search Model

Implement search around the original acquisition date.

Default target window:

approximately `±30 days`

but make it configurable/tunable rather than embedding the assumption everywhere.

Candidates must:

- overlap the relevant geographic footprint
- be Sentinel-1 GRD
- be on another acquisition date
- contain enough metadata to establish they are genuinely different products

Prefer comparable acquisitions where possible.

Do not force a match merely because something intersects the bounding box.

### Phase C — No-Match Behavior

If no suitable comparison exists:

return:

`no comparison available`

or an equivalent explicit status.

That is a valid result.

### Phase D — Fetch + Inference

For selected comparison observations:

reuse the existing Sentinel processing and inference path.

Preserve for each result:

- real product/scene identity
- acquisition date
- source metadata
- detection/inference result

Ensure the original and comparison images cannot accidentally reference the same cached payload.

### Phase E — UI/API

Expose:

- original observation
- comparison observation(s)
- dates for each
- detection result for each

Provide side-by-side visual comparison where the existing frontend permits it.

Do NOT implement automated change classification.

### Phase F — Verification

Tests should cover:

- candidate-date filtering
- same-scene exclusion
- no comparison available
- valid alternate pass
- fetch failure
- inference failure
- serialization
- frontend/API handling

When real network credentials are available, test at least one real location for which a real second pass exists.

Verify using product metadata that the comparison is actually another observation.

### TASK 2 ACCEPTANCE CRITERIA

Complete when:

- existing auth/fetch/inference code is reused
- a genuine different acquisition can be selected
- same/cached scene duplication is guarded against
- no-match behavior is honest
- results retain real dates
- UI/API supports comparisons
- no automated persistence verdict is introduced
- tests pass
- limitations are documented in `docs/status.md`

Then review the diff before Task 3.

---

# TASK 3 — SENTINEL-2 OPTICAL SUPPLEMENT

Priority: **MEDIUM**

## OBJECTIVE

Provide a real Sentinel-2 L2A true-color optical scene covering approximately the same area near the Sentinel-1 acquisition date when a sufficiently low-cloud scene exists.

This is a VISUAL SUPPLEMENT.

It is NOT image fusion.

It is NOT cloud removal.

It is NOT automated SAR-optical classification.

---

## TASK 3 LOOP

### Phase A — Reuse Existing Infrastructure

Inspect existing CDSE clients before adding anything.

Reuse authentication/configuration where possible.

Determine the cleanest place for Sentinel-2 catalog and rendering logic.

### Phase B — Search

Given:

- Sentinel-1 bounding box
- acquisition date

search for Sentinel-2 L2A candidates near that date.

Use the appropriate cloud-cover metadata supported by the actual API/catalog.

Do not assume a metadata field name without verifying the response schema.

Prefer:

1. geographic coverage
2. low cloud
3. temporal proximity

Keep the actual optical acquisition date.

### Phase C — Cloud Threshold

Use a configurable threshold.

If no candidate satisfies it:

return:

`no cloud-free optical scene available`

Do not quietly select a bad scene.

### Phase D — Rendering

Fetch a true-color RGB rendering cropped approximately to the same geographic region.

Avoid unnecessary ML or geospatial complexity.

### Phase E — UI/API

Display optical and SAR data side-by-side.

Clearly label:

- SAR acquisition date
- Sentinel-2 acquisition date

Never imply they are same-day unless they truly are.

### Phase F — Verification

Test:

- candidate filtering
- cloud threshold
- no-match response
- valid candidate selection
- date preservation
- image/render request
- API response
- UI

If external access works, visually sanity-check one real pair to confirm both cover the intended area.

### TASK 3 ACCEPTANCE CRITERIA

Complete when:

- real Sentinel-2 L2A data is used
- cloud filtering works
- no-match behavior is explicit
- dates are honest
- no fusion/cloud-removal code was added
- UI/API integration works
- tests pass
- `docs/status.md` records implementation and limitations

Then review the full diff.

---

# TASK 4 — DSen2-CR CLOUD REMOVAL EXPERIMENT

Priority: **LOW / OPTIONAL**

Only start after Tasks 1–3 are complete or genuinely blocked.

This task has a much stricter stop policy.

Known risk:

- original implementation relies on old TensorFlow/Keras tooling
- Python 3.7 / TensorFlow 1.15 era dependencies
- PyTorch port may be incomplete
- reproducibility is questionable

Do not destabilize Pelagic for this experiment.

---

# TASK 4 ISOLATION RULE

No DSen2-CR dependencies may be introduced into:

- main Pelagic virtual environment
- main requirements
- working application runtime

Use a completely isolated environment such as:

- Docker container
- isolated legacy venv

The experiment must be disposable.

---

# TASK 4 LOOP

## CHECKPOINT 1 — VIABILITY ONLY

First determine whether the official implementation can:

1. install in isolation
2. import successfully
3. instantiate the model
4. load official pretrained weights

Do not start integration work before these succeed.

### HARD STOP

If the official model cannot load pretrained weights in the isolated environment after a reasonable bounded investigation:

STOP TASK 4.

Document:

- environment attempted
- dependency versions
- exact failure
- relevant commands
- whether failure appears reproducible
- recommendation

Do not continue indefinitely.

## CHECKPOINT 2 — ONE REAL INFERENCE

Only after Checkpoint 1 passes:

obtain one real Sentinel-1 / Sentinel-2 pair through the existing project data access path.

Attempt one inference.

Evaluate:

- pipeline completes
- output dimensions/data are sensible
- output is visually plausible

Do not claim scientifically valid cloud removal merely because code runs.

## CHECKPOINT 3 — REPORT ONLY

Even if successful:

DO NOT integrate DSen2-CR into the live application.

Only report findings and update documentation.

Integration requires a later explicit decision.

---

# 3. TESTING PHILOSOPHY

Prefer a pyramid:

### Unit
Test deterministic logic:

- wind magnitude
- candidate ranking
- date filtering
- cloud threshold
- status conversion

### Integration
Test module boundaries:

- ERA5 adapter
- CDSE catalog client
- live-fetch supplementary analysis
- API response

### Real external test
Run only when credentials/network permit.

Never make standard CI depend on external satellite APIs unless the repository explicitly already does this.

Separate:

`unit pass`

from:

`real API verified`

in documentation.

---

# 4. ERROR HANDLING

External services WILL fail.

Handle at minimum:

- credentials missing
- invalid credentials
- request rejected
- request timeout
- no catalog result
- malformed metadata
- unavailable dataset
- download/render failure
- inference failure

Optional supplementary analysis failure should generally produce a structured status rather than destroy the primary live detection response.

Never silently swallow errors.

Never expose secrets in logs.

---

# 5. DOCUMENTATION LOOP

After EACH completed or blocked task, update:

`docs/status.md`

Record:

- what was attempted
- what was implemented
- what was actually tested
- what was externally verified
- what was skipped
- known limitations
- remaining work

Use precise language.

Examples:

GOOD:
`ERA5 integration implemented and unit-tested. Live CDS verification skipped because CDSAPI_KEY was not configured.`

BAD:
`ERA5 integration complete.`

if no real ERA5 request was ever made.

---

# 6. CHANGE CONTROL

Throughout the work:

- inspect `git diff`
- keep changes task-focused
- avoid unrelated formatting churn
- avoid dependency upgrades unless needed
- avoid broad refactors during feature work
- do not delete existing functionality without evidence
- preserve backwards compatibility when practical

If a refactor becomes necessary:

first demonstrate why the current abstraction prevents the feature.

---

# 7. RECOVERY LOOP

Whenever an implementation fails:

DO NOT immediately try random fixes.

Use:

**OBSERVE → LOCALIZE → FORM HYPOTHESIS → TEST HYPOTHESIS → FIX → RETEST**

Specifically:

1. capture the real error
2. identify the failing boundary
3. inspect relevant code/documentation
4. form one likely explanation
5. make the smallest change capable of disproving/fixing it
6. rerun the failing test
7. only then continue

Avoid dependency roulette.

Avoid repeatedly adding packages until an error disappears.

---

# 8. DEFINITION OF DONE

The overall assignment is done when:

## Track 1
ERA5 wind supplementary evidence works or has a documented genuine blocker.

## Track 2
Multi-temporal Sentinel-1 comparison works or has a documented genuine blocker.

## Track 3
Sentinel-2 low-cloud optical supplement works or has a documented genuine blocker.

## Track 4
Either:
- isolated DSen2-CR viability was demonstrated and reported, OR
- the viability checkpoint failed and the failure was documented.

Task 4 is NOT required to succeed.

Most importantly:

the original live Sentinel-1 detection pipeline must remain functional.

---

# 9. FINAL REPOSITORY REVIEW

After all reachable tasks are complete:

run one final loop.

## Inspect
Review:

- git diff
- test results
- config changes
- docs
- API schemas
- frontend assumptions

## Verify

Run the broadest practical local test suite.

Check that no:

- credential
- secret
- downloaded giant dataset
- temporary artifact
- generated satellite file
- experimental DSen2 environment

was accidentally committed.

## Scientific-integrity audit

Search new code for suspicious fallbacks such as:

- hardcoded wind speeds
- fake timestamps
- placeholder coordinates
- sample satellite values being returned as production results
- automatic oil verdicts
- fallback comparison images

Remove them unless explicitly test-only fixtures.

---

# 10. FINAL REPORT FORMAT

When finished, report:

## Implemented

For each task, explain only what actually exists.

## Verification

Separate:

- automated/local tests
- real external API verification

## Blocked / Skipped

State exact reasons.

## Files Changed

List important files and what changed.

## Tests

Give commands run and their outcomes.

## External Data Verified

List real external services/scenes tested, if any.

If none, say so explicitly.

## Limitations

Include scientific and engineering limitations.

## Recommended Next Step

Give the single highest-value next action based on the actual state of the repository.

Do not exaggerate completion.

---

# EXECUTION START

Begin now.

First inspect the repository and reconstruct the real live-fetch architecture.

Do not start coding until you understand where:

- CDSE authentication lives
- Sentinel-1 fetching lives
- inference is invoked
- detection metadata originates
- API results are constructed
- the frontend consumes those results

Then execute **Task 1** using the engineering loop.

After Task 1 satisfies its acceptance criteria or reaches a genuine documented blocker, proceed automatically to Task 2, then Task 3, and only then consider Task 4.

Do not stop merely to ask permission between normal implementation iterations.

Stop only for:

- a genuine requirement requiring human choice
- unavailable mandatory credentials that cannot be bypassed with honest testing
- a destructive action requiring approval
- the explicit Task 4 hard-stop condition
- completion of all reachable work.
