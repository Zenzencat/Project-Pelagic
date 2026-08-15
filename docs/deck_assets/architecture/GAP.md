# No rendered architecture visual exists

Checked before assuming otherwise: `docs/system_architecture.md` contains a system architecture diagram written as a **Mermaid code block** (markdown text), not a rendered PNG/SVG/JPG image file anywhere in the repo. No other file in the repository contains an architecture-related image either — confirmed by a full repo-wide image inventory (see `docs/deck_assets/MANIFEST.md`).

**This is a real gap for the deck.** If you want an architecture slide, you'll need to either:
- Render the Mermaid block from `docs/system_architecture.md` (any Mermaid renderer, e.g. the Mermaid Live Editor, or a VS Code Mermaid preview extension) and export it as an image, or
- Build the slide directly from the corrected architecture description in that file (it was fixed in an earlier round — the diagram no longer claims ONNX Runtime / a ResNet backbone, matching the real stack).

Not generated this round — this round was inventory/consolidation only, no new image generation, per explicit instruction.
