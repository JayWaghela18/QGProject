"""
inspect_model.py
----------------
Loads HR_CNN_Time_Frequency_v3.keras (using the normal venv) and:
  1. Prints a full layer-by-layer breakdown with param counts
  2. Saves the architecture as model_architecture.json
  3. Prints quick stats (inputs, outputs, total params)
"""

import os
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# Suppress absl logging
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

MODEL_PATH = "models/HR_CNN_Time_Frequency_v3.keras"
JSON_OUT   = "model_architecture.json"

print("Loading model ...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!\n")

# ── 1. Quick stats ────────────────────────────────────────────
print("=" * 65)
print("  QUICK STATS")
print("=" * 65)
print(f"  Model class   : {model.__class__.__name__}")
print(f"  TF version    : {tf.__version__}")
print(f"  Total layers  : {len(model.layers)}")
print(f"  Total params  : {model.count_params():,}")
print(f"  Inputs  ({len(model.inputs)})  :", [(inp.name, tuple(inp.shape)) for inp in model.inputs])
print(f"  Outputs ({len(model.outputs)}) :", [(out.name, tuple(out.shape)) for out in model.outputs])

# ── 2. Layer-by-layer breakdown ───────────────────────────────
print("\n" + "=" * 65)
print("  LAYER-BY-LAYER BREAKDOWN")
print("=" * 65)
print(f"  {'#':<5} {'Type':<28} {'Name':<32} {'Params':>10}")
print("  " + "-" * 80)
for i, layer in enumerate(model.layers):
    try:
        params = layer.count_params()
    except Exception:
        params = -1
    pstr = f"{params:,}" if params >= 0 else "N/A"
    print(f"  [{i:>3}]  {layer.__class__.__name__:<28} {layer.name:<32} {pstr:>10}")

# ── 3. Save JSON architecture ─────────────────────────────────
print(f"\n{'=' * 65}")
print(f"  Saving architecture to '{JSON_OUT}' ...")
arch = json.loads(model.to_json())
with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(arch, f, indent=2)
print(f"  Saved! Open '{JSON_OUT}' in VS Code to browse the full JSON.")

# ── 4. Netron hint ────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("  VISUAL GRAPH → Netron")
print("=" * 65)
print("  1. Visit  https://netron.app")
print("  2. Click  'Open Model'")
print("  3. Select  models/HR_CNN_Time_Frequency_v3.keras")
print("=" * 65)
print("\nDone.")
