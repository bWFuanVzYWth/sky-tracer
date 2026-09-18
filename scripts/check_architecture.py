"""Check solver ownership without importing or running any solver."""
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]
CORES = {"sky-pt", "sky-reference", "sky-realtime"}
TABLES = {"atmosphere_profile.csv", "aerosol_profile.csv", "aerosol_optics.csv",
          "mie_phase.csv", "bands.csv", "CIE_xyz_1931_2deg.csv"}
workspace = tomllib.loads((ROOT / "Cargo.toml").read_text())
members = workspace["workspace"]["members"]
assert set(p.name for p in (ROOT / "crates").iterdir() if (p / "Cargo.toml").exists()) == CORES | {"sky-assets"}
for name in sorted(CORES):
    core = ROOT / "crates" / name
    manifest = tomllib.loads((core / "Cargo.toml").read_text())
    for section in ("dependencies", "dev-dependencies", "build-dependencies"):
        for dep, spec in manifest.get(section, {}).items():
            assert not dep.startswith("sky-"), (name, dep)
            assert not isinstance(spec, dict) or "path" not in spec, (name, dep)
    assert not manifest.get("target"), "Review target-specific dependencies explicitly"
    for filename in TABLES:
        path = core / "data" / filename
        assert path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1, path
    for path in core.rglob("*.rs"):
        text = path.read_text(encoding="utf-8")
        for other in (CORES | {'sky-assets'}) - {name}:
            assert other.replace('-', '_') + '::' not in text, (path, other)
        assert not re.search(r'(?:include_str!|include_bytes!)\([^\n]*out[/\\]', text), path
    print(f"{name}: independent crate and owned tables")
assert not (ROOT / "data").exists(), "No shared root input directory"
for path in (ROOT / "apps").rglob("*.wgsl"):
    assert path.parent.name == "shaders" and path.name in {
        "fullscreen_debug.wgsl", "hdr_ui_composite.wgsl", "present_texture.wgsl", "reinhard_gamut.wgsl"
    }, f"Atmospheric shader belongs in a solver: {path}"
for member in members:
    assert (ROOT / member / "Cargo.toml").is_file(), member
print(f"Architecture OK: {len(CORES)} independent solvers, {len(members)} Rust packages")
