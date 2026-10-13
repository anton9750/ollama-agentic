
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path.cwd()

datas = []

for folder in ("skills", "workflows"):
    path = ROOT / folder
    if path.exists():
        datas.append((str(path), folder))

for filename in ("team-config.json",):
    path = ROOT / filename
    if path.exists():
        datas.append((str(path), "."))

hiddenimports = collect_submodules("team")

a = Analysis(
    ["launcher.py"],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AgenticAICompanion",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="AgenticAICompanion",
)
