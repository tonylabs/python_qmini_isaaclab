# CLAUDE.md

## ONNX export

- Always export trained policies to the **project root** with the file name **`policy.onnx`**
  (i.e. `<repo>/policy.onnx`). Never export to a run/log subfolder or under another name —
  downstream tooling (`scripts/sim2sim.py`, the robot deployment runtime) loads exactly this path.
- `scripts/export_onnx.py` already defaults to this destination; keep it that way and preserve
  that behavior in any new export script.
