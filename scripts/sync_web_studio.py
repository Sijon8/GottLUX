"""Regenerate the look-spec block in web/js/looks.js from gottlux.studio (run after changing a
look's parameters; tests/test_studio.py fails while the two differ)."""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gottlux.studio.looks import look_specs  # noqa: E402

JS = os.path.join(ROOT, "web", "js", "looks.js")
BLOCK = re.compile(r"(/\*SPECS-BEGIN\*/).*?(/\*SPECS-END\*/)", re.S)


def render() -> str:
    import numpy as np

    from gottlux.studio.looks import COLORMAPS, lut
    from gottlux.studio.shot import PALETTES, PRESETS
    stops = np.linspace(0, 255, 17).astype(int)
    data = {"looks": look_specs(), "palettes": PALETTES,
            "presets": {k: list(v) for k, v in PRESETS.items()},
            "colormaps": {n: np.round(lut(n)[stops], 4).tolist() for n in COLORMAPS}}
    return "/*SPECS-BEGIN*/\n  var SPEC = " + json.dumps(data, indent=1, ensure_ascii=False) \
        + ";\n  /*SPECS-END*/"


def main():
    with open(JS, encoding="utf-8") as f:
        src = f.read()
    new = BLOCK.sub(lambda m: render(), src)
    with open(JS, "w", encoding="utf-8", newline="\n") as f:
        f.write(new)
    print("updated", JS)


if __name__ == "__main__":
    main()
