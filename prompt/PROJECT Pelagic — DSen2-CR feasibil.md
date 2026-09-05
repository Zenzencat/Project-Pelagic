PROJECT: Pelagic — DSen2-CR feasibility attempt via Kaggle (Task 4, scoped)

CONTEXT
This was already investigated once and closed by explicit decision
(commit 4e0b670) rather than attempted, per the brief's own risk
warning: the maintainer's README admits their own results aren't
reproducible, the available PyTorch port imports a nonexistent
base_model.BaseModel class with no pretrained weights, and genuine
working usage requires Keras/TensorFlow 1.15 + Python 3.7 — all
end-of-life. This is still HIGH-RISK and LOW-PRIORITY relative to
everything else in this project. Only attempt this because there's
spare time and a spare Kaggle T4 GPU; do not let it block or
destabilize anything in the working live-fetch pipeline, and do not
touch the main project venv/requirements at all.

GOAL
Use a Kaggle kernel as the isolated environment (instead of a local
Docker container) to check whether DSen2-CR's official checkpoints
can actually be loaded and run at all. This is a feasibility check,
not an integration attempt.

PHASE 1 — Environment feasibility (STOP here if this fails)
- Create a new Kaggle kernel (kaggle_kernel_dsen2cr/, matching the
  existing kaggle_kernel/ and kaggle_kernel_lookalike/ convention in
  this repo) with GPU T4 enabled via kernel-metadata.json
  (machine_shape: NvidiaTeslaT4 — the same setting this project
  already had to discover was required, see AGENTS.md/status.md
  history for why enable_gpu: true alone is not enough).
- Attempt to install TensorFlow 1.15 + the exact Python version
  DSen2-CR's official (non-PyTorch-port) implementation requires,
  inside that kernel. Kaggle's current base images may not support
  Python 3.7 at all — check this FIRST before installing anything
  else, since it may make the whole approach a dead end immediately.
- If Python 3.7/TF 1.15 cannot be installed in a Kaggle kernel:
  STOP. Report exactly what was tried and what failed. Do not attempt
  workarounds like force-installing incompatible wheels or patching
  the library — that's exactly the rabbit hole the original brief
  warned against.
- If it CAN be installed: confirm the official pretrained checkpoints
  (already verified byte-level downloadable, per the original
  investigation) actually load without error in that environment.
  This is checkpoint 1 — if loading fails, STOP and report here too.

PHASE 2 — Single real inference attempt (only if Phase 1 fully passes)
- Fetch one real Sentinel-1 + Sentinel-2 pair (same area/time) using
  this project's existing CDSE credentials — do not use synthetic or
  placeholder data.
- Run one inference pass through the loaded DSen2-CR model.
- Report plainly whether the output looks like a plausible
  cloud-removed reconstruction, or garbage/noise/an obvious failure —
  a visual sanity check, not a formal metric.

HARD LIMITS
- Complete isolation: the Kaggle kernel's environment must not touch
  or modify this project's main requirements.txt, main venv, or any
  file outside kaggle_kernel_dsen2cr/.
- Do NOT integrate this into the live-fetch pipeline even if it works
  — this task ends at "does it work," not "wire it in."
- Time-box it: if Phase 1 alone isn't resolved within one Kaggle
  session, stop and report what was tried and where it failed rather
  than continuing to debug across multiple sessions.
- Use the existing Kaggle API credentials already working in this
  project (trongzen account) — do not set up new ones.

REPORT BACK
- Phase 1 result: did Python 3.7/TF 1.15 install on Kaggle at all?
  Did the checkpoint load?
- Phase 2 result (if reached): does inference produce a plausible
  cloud-removed image?
- A clear go/no-go recommendation on whether this is worth further
  time, independent of whether the answer is "yes it works" or
  "no, dead end confirmed for a second time."