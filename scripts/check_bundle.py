"""
check_bundle.py — verify an offline wheel bundle matches the interpreter running it.

pip reports an incompatible bundle as "no matching distribution found … (from
versions: none)", which does not say *why*. This preflight compares the wheel tags in
``vendor/wheels`` against the running interpreter and reports the exact mismatch:
wrong Python version, wrong operating system, or an empty/absent bundle.

    python scripts/check_bundle.py             # checks ./vendor/wheels
    python scripts/check_bundle.py <folder>

Exit status is 0 when the bundle is usable, 1 otherwise, so installer scripts can gate
on it.
"""
from __future__ import annotations

import os
import platform
import sys

#: A wheel filename is  name-version(-build)?-pytag-abitag-platformtag.whl
def wheel_tags(filename: str):
    """The (python, abi, platform) tag triple of a wheel filename."""
    stem = filename[:-4] if filename.lower().endswith(".whl") else filename
    parts = stem.split("-")
    if len(parts) < 5:
        return None
    return parts[-3], parts[-2], parts[-1]


def interpreter_tags():
    """The tag pieces this interpreter can install: (cp-tag, os-fragment)."""
    cp = f"cp{sys.version_info.major}{sys.version_info.minor}"
    system = platform.system().lower()
    if system == "windows":
        osfrag = "win"
    elif system == "linux":
        osfrag = "linux"
    elif system == "darwin":
        osfrag = "macos"
    else:
        osfrag = system
    return cp, osfrag


def compatible(tags, cp, osfrag) -> bool:
    """True when a wheel's tags plausibly match this interpreter.

    Pure-Python wheels (``py3-none-any``) fit anywhere; extension wheels must match both
    the CPython version and the operating system.
    """
    pytag, _abi, plat = tags
    if plat == "any":
        return True
    if osfrag not in plat.lower():
        return False
    return cp in pytag or pytag.startswith("py3")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = argv[0] if argv else os.path.join(root, "vendor", "wheels")

    cp, osfrag = interpreter_tags()
    here = (f"Python {sys.version_info.major}.{sys.version_info.minor} "
            f"on {platform.system()} ({platform.machine()})")

    if not os.path.isdir(folder):
        print(f"[!] No wheel bundle at: {folder}")
        print("    A clone of the repository does not include one (it is gitignored).")
        print("    Build it on a networked machine with:  python scripts/fetch_wheels.py")
        print("    then copy the whole vendor/ folder across.")
        return 1

    wheels = [f for f in os.listdir(folder) if f.lower().endswith(".whl")]
    if not wheels:
        print(f"[!] The bundle folder is empty: {folder}")
        print("    Rebuild with:  python scripts/fetch_wheels.py")
        return 1

    usable, foreign = [], []
    for w in wheels:
        tags = wheel_tags(w)
        (usable if tags and compatible(tags, cp, osfrag) else foreign).append(w)

    # Extension wheels (not pure-Python) are the ones that must match.
    native = [w for w in usable if wheel_tags(w) and wheel_tags(w)[2] != "any"]

    print(f"bundle : {folder}")
    print(f"wheels : {len(wheels)}  ({len(usable)} usable here, {len(foreign)} for other targets)")
    print(f"running: {here}")

    info = os.path.join(folder, "BUNDLE_INFO.txt")
    if os.path.exists(info):
        with open(info, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(("target Python", "target platform")):
                    print("bundle " + line.rstrip())

    if not native:
        print()
        print("[!] This bundle does not match this machine.")
        print(f"    Not one compiled wheel targets {here}.")
        example = next((w for w in foreign if wheel_tags(w) and wheel_tags(w)[2] != "any"), None)
        if example:
            t = wheel_tags(example)
            print(f"    Example wheel in the bundle: {example}")
            print(f"    (targets {t[0]} / {t[2]}, this interpreter needs {cp} / *{osfrag}*)")
        print()
        print("    Fix: rebuild the bundle for this machine, on a networked computer:")
        print(f"        python scripts/fetch_wheels.py --python-version "
              f"{sys.version_info.major}.{sys.version_info.minor}")
        if osfrag != "win":
            print("        (add --platform manylinux2014_x86_64 for a Linux target)")
        print("    Windows note: run install_offline.bat, not the .sh script.")
        return 1

    print("\nBundle is compatible with this interpreter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
