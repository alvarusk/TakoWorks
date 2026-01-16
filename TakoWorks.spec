# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
import os

datas = [('data', 'data'), ('bin', 'bin'), ('assets', 'assets'), ('src\\takoworks\\modules\\transcriber\\weight', 'takoworks\\modules\\transcriber\\weight'), ('src\\takoworks\\modules\\transcriber\\weight', 'takoworks\\modules\\transcriber\\weight')]
binaries = []
hiddenimports = []
tmp_ret = collect_all('pykakasi')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('unidic_lite')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
try:
    import unidic_lite
    dicdir_path = os.path.join(os.path.dirname(unidic_lite.__file__), "dicdir")
    if os.path.isdir(dicdir_path):
        has_dicdir = False
        for _src, _dest in datas:
            if _dest.replace("\\", "/").endswith("unidic_lite/dicdir"):
                has_dicdir = True
                break
        if not has_dicdir:
            datas.append((dicdir_path, os.path.join("unidic_lite", "dicdir")))
except Exception:
    pass


a = Analysis(
    ['run_takoworks.py'],
    pathex=['.\\src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TakoWorks',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets\\takoworks_big.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TakoWorks',
)
