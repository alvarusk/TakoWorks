from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import tkinter as tk
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ... import __version__, paths
from ...config import save_config


def _add_to_path(*dirs: str) -> None:
    current = os.environ.get("PATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    for d in dirs:
        if d and d not in parts:
            parts.insert(0, d)
    os.environ["PATH"] = os.pathsep.join(parts)


GITHUB_OWNER = "alvarusk"
GITHUB_REPO = "takoworks"
WORKFLOW_FILE = "release.yml"
ARTIFACT_NAME = "takoworks-windows"
ARTIFACT_ZIP = "TakoWorks_win64.zip"
RAW_VERSION_URL = (
    f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/main/src/takoworks/__init__.py"
)
USER_AGENT = "TakoWorks-Update"


class SettingsPanel(ttk.Frame):
    def __init__(self, parent, runner, cfg: dict):
        super().__init__(parent)
        self.runner = runner
        self.cfg = cfg

        self.jpdict_var = tk.StringVar(value=cfg.get("jpdict_dir", ""))
        self.cndict_var = tk.StringVar(value=cfg.get("cndict_dir", ""))
        self.ffmpeg_var = tk.StringVar(value=cfg.get("ffmpeg_dir", ""))
        self.yomitoku_var = tk.StringVar(value=cfg.get("yomitoku_dir", ""))

        self._build()

    def _build(self):
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(frm, text="Paths (stored in config.json)").pack(anchor="w", pady=(0, 8))

        self._row(frm, "Japanese Dictionary Folder", self.jpdict_var, self._pick_dir)
        self._row(frm, "SChinese Dictionary Folder", self.cndict_var, self._pick_dir)
        self._row(frm, "FFMPEG Folder (ffmpeg.exe)", self.ffmpeg_var, self._pick_dir)
        self._row(frm, "YomiToku Folder (yomitoku.exe)", self.yomitoku_var, self._pick_dir)

        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=10)

        ttk.Button(btns, text="Save & Apply", command=self._save_apply).pack(side="left")
        ttk.Button(btns, text="Apply & Don't Save", command=self._apply_only).pack(side="left", padx=6)

        ttk.Separator(frm).pack(fill="x", pady=10)

        ttk.Label(frm, text="Notes:").pack(anchor="w")
        ttk.Label(
            frm,
            text="- Transcriber reads YOMI_JA_DIR/YOMI_ZH_DIR when importing.\n"
            "- As it's imported when run, changing here before the run works.\n"
            "- PATH is applied to sub-processes (ffmpeg/yomitoku).",
            justify="left",
        ).pack(anchor="w", pady=6)

        ttk.Separator(frm).pack(fill="x", pady=10)

        ttk.Label(frm, text=f"Updates (current v{__version__})").pack(anchor="w")
        ttk.Button(frm, text="Actualizar (Actions)", command=self._check_updates).pack(anchor="w", pady=4)

    def _row(self, parent, label, var, browse_cmd):
        r = ttk.Frame(parent)
        r.pack(fill="x", pady=3)
        ttk.Label(r, text=label).pack(side="left")
        ttk.Entry(r, textvariable=var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r, text="Browse", command=lambda: browse_cmd(var)).pack(side="left")

    def _pick_dir(self, var):
        p = filedialog.askdirectory()
        if p:
            var.set(p)

    def _apply_only(self):
        self._apply_env()

    def _save_apply(self):
        self.cfg["jpdict_dir"] = self.jpdict_var.get().strip()
        self.cfg["cndict_dir"] = self.cndict_var.get().strip()
        self.cfg["ffmpeg_dir"] = self.ffmpeg_var.get().strip()
        self.cfg["yomitoku_dir"] = self.yomitoku_var.get().strip()

        save_config(self.cfg)
        self._apply_env()
        messagebox.showinfo("OK", "Settings saved and applied.")

    def _apply_env(self):
        jp = self.jpdict_var.get().strip()
        zh = self.cndict_var.get().strip()
        ff = self.ffmpeg_var.get().strip()
        yo = self.yomitoku_var.get().strip()

        if jp:
            os.environ["YOMI_JA_DIR"] = jp
        if zh:
            os.environ["YOMI_ZH_DIR"] = zh

        _add_to_path(ff, yo)

    def _check_updates(self):
        if not paths.is_frozen():
            messagebox.showinfo(
                "Actualizar",
                "Este boton es para la version empaquetada. En desarrollo, usa git pull.",
            )
            return

        token = os.environ.get("GITHUB_TOKEN", "").strip() or None

        try:
            remote_version = self._fetch_remote_version(token)
        except Exception as e:
            messagebox.showerror("Actualizar", f"No pude leer la version remota.\n\n{e}")
            return

        if remote_version == __version__:
            messagebox.showinfo("Actualizar", f"Ya estas al dia (v{__version__}).")
            return

        if not messagebox.askyesno(
            "Actualizar",
            f"Version nueva detectada (local v{__version__} -> remota v{remote_version}).\n\n"
            "Descargar desde GitHub Actions e instalar ahora? Esto cerrara TakoWorks y lo relanzara.",
        ):
            return

        try:
            run_info = self._latest_actions_artifact(token)
        except Exception as e:
            messagebox.showerror(
                "Actualizar",
                "No pude localizar el build mas reciente en GitHub Actions.\n\n"
                f"{e}\n\nComprueba que el workflow {WORKFLOW_FILE} haya pasado y usa GITHUB_TOKEN.",
            )
            return

        try:
            payload_dir, temp_root = self._download_artifact(run_info, token)
        except Exception as e:
            messagebox.showerror("Actualizar", f"Fallo al descargar el artefacto.\n\n{e}")
            return

        try:
            self._schedule_apply(payload_dir, temp_root, run_info, remote_version)
        except Exception as e:
            messagebox.showerror("Actualizar", f"Fallo al preparar la actualizacion.\n\n{e}")

    def _headers(self, token: str | None = None) -> dict:
        headers = {"User-Agent": USER_AGENT}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _fetch_remote_version(self, token: str | None) -> str:
        req = urllib.request.Request(RAW_VERSION_URL, headers=self._headers(token))
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status} al leer la version remota.")
            text = resp.read().decode("utf-8")
        match = re.search(r'__version__\\s*=\\s*"([^"]+)"', text)
        if not match:
            raise RuntimeError("No encontre __version__ en main.")
        return match.group(1)

    def _latest_actions_artifact(self, token: str | None) -> dict:
        runs_url = (
            f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/workflows/"
            f"{WORKFLOW_FILE}/runs?branch=main&status=success&per_page=1"
        )
        req = urllib.request.Request(runs_url, headers=self._headers(token))
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        runs = data.get("workflow_runs") or []
        if not runs:
            raise RuntimeError("No hay runs exitosos del workflow.")
        run = runs[0]
        run_id = run.get("id")
        run_number = run.get("run_number")

        artifacts_url = run.get("artifacts_url") or (
            f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/runs/{run_id}/artifacts"
        )
        req = urllib.request.Request(artifacts_url, headers=self._headers(token))
        with urllib.request.urlopen(req, timeout=30) as resp:
            art_data = json.loads(resp.read().decode("utf-8"))
        artifacts = art_data.get("artifacts") or []
        if not artifacts:
            raise RuntimeError("No hay artefactos en el ultimo run.")
        artifact = next((a for a in artifacts if a.get("name") == ARTIFACT_NAME), artifacts[0])
        download_url = artifact.get("archive_download_url")
        if not download_url:
            raise RuntimeError("No encontre archive_download_url.")
        return {
            "run_id": run_id,
            "run_number": run_number,
            "download_url": download_url,
            "html_url": run.get("html_url"),
        }

    def _download_artifact(self, run_info: dict, token: str | None):
        temp_root = Path(tempfile.mkdtemp(prefix="takoworks_update_"))
        artifact_zip = temp_root / "artifact.zip"
        req = urllib.request.Request(run_info["download_url"], headers=self._headers(token))
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, open(artifact_zip, "wb") as fh:
                shutil.copyfileobj(resp, fh)
        except urllib.error.HTTPError as e:
            msg = f"HTTP {e.code}"
            if e.code == 404:
                msg = "404 (necesitas GITHUB_TOKEN para bajar artefactos)"
            raise RuntimeError(msg)

        with zipfile.ZipFile(artifact_zip) as zf:
            zf.extractall(temp_root)

        inner_zip = None
        for candidate in temp_root.rglob("*.zip"):
            if candidate.name == ARTIFACT_ZIP:
                inner_zip = candidate
                break
        if not inner_zip:
            raise RuntimeError(f"No encontre {ARTIFACT_ZIP} dentro del artefacto.")

        payload_dir = temp_root / "payload"
        payload_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(inner_zip) as zf:
            zf.extractall(payload_dir)

        self.runner._console_write(
            f"[Update] Descargado run #{run_info.get('run_number')} (id {run_info.get('run_id')}) a {payload_dir}"
        )
        return payload_dir, temp_root

    def _schedule_apply(self, payload_dir: Path, temp_root: Path, run_info: dict, remote_version: str):
        install_dir = paths.app_root()
        ps1 = temp_root / "apply_update.ps1"
        script = f"""param(
    [string]$InstallDir,
    [string]$PayloadDir,
    [int]$ProcId
)
$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyyMMddHHmmss"
$backup = "$InstallDir.bak.$timestamp"
while (Get-Process -Id $ProcId -ErrorAction SilentlyContinue) {{ Start-Sleep -Seconds 1 }}
if (Test-Path $backup) {{ Remove-Item -Recurse -Force $backup }}
if (Test-Path $InstallDir) {{
    Move-Item -Path $InstallDir -Destination $backup -Force
}}
New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
Copy-Item -Path (Join-Path $PayloadDir '*') -Destination $InstallDir -Recurse -Force
Start-Process (Join-Path $InstallDir 'TakoWorks.exe')
"""
        ps1.write_text(script, encoding="utf-8")

        flags = 0
        if hasattr(subprocess, "DETACHED_PROCESS"):
            flags |= subprocess.DETACHED_PROCESS
        if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            flags |= subprocess.CREATE_NEW_PROCESS_GROUP

        cmd = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ps1),
            "-InstallDir",
            str(install_dir),
            "-PayloadDir",
            str(payload_dir),
            "-ProcId",
            str(os.getpid()),
        ]
        subprocess.Popen(cmd, creationflags=flags)

        self.cfg.setdefault("updates", {})["pending_version"] = remote_version
        save_config(self.cfg)
        messagebox.showinfo(
            "Actualizar",
            "Descarga completada. Se cerrara TakoWorks para aplicar la actualizacion y se relanzara.",
        )
        self.winfo_toplevel().after(300, self.winfo_toplevel().destroy)
