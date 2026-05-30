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
from ...config import save_config, save_local_config


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

        self.ffmpeg_var = tk.StringVar(value=cfg.get("ffmpeg_dir", ""))
        self.deepl_var = tk.StringVar(value=cfg.get("api_keys", {}).get("deepl", ""))

        self._build()

    def _build(self):
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(frm, text="Paths (stored in config.json)").pack(anchor="w", pady=(0, 8))

        self._row(frm, "FFmpeg Folder (ffmpeg.exe)", self.ffmpeg_var, self._pick_dir)

        ttk.Separator(frm).pack(fill="x", pady=10)

        ttk.Label(frm, text="API Keys").pack(anchor="w", pady=(0, 8))
        self._row(frm, "DeepL Auth Key", self.deepl_var, None, secret=True)

        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=10)

        ttk.Button(btns, text="Save & Apply", command=self._save_apply).pack(side="left")
        ttk.Button(btns, text="Apply & Don't Save", command=self._apply_only).pack(side="left", padx=6)

        ttk.Separator(frm).pack(fill="x", pady=10)

        ttk.Label(frm, text="Notes:").pack(anchor="w")
        ttk.Label(
            frm,
            text="- FFmpeg is added to subprocesses via PATH.\n"
            "- DeepL API Free keys end in ':fx' and automatically use api-free.deepl.com.\n"
            "- The app also keeps its bundled tools on PATH when available.",
            justify="left",
        ).pack(anchor="w", pady=6)

    def _row(self, parent, label, var, browse_cmd, secret: bool = False):
        r = ttk.Frame(parent)
        r.pack(fill="x", pady=3)
        ttk.Label(r, text=label).pack(side="left")
        entry = ttk.Entry(r, textvariable=var, show="*" if secret else "")
        entry.pack(side="left", fill="x", expand=True, padx=6)
        if browse_cmd is not None:
            ttk.Button(r, text="Browse", command=lambda: browse_cmd(var)).pack(side="left")

    def _pick_dir(self, var):
        p = filedialog.askdirectory()
        if p:
            var.set(p)

    def _apply_only(self):
        self._apply_env()

    def _save_apply(self):
        self.cfg["ffmpeg_dir"] = self.ffmpeg_var.get().strip()
        deepl_key = self.deepl_var.get().strip()

        save_config(self.cfg)
        save_local_config({"api_keys": {"deepl": deepl_key}})
        self._apply_env()
        messagebox.showinfo("OK", "Settings saved and applied.")

    def _apply_env(self):
        ff = self.ffmpeg_var.get().strip()

        _add_to_path(ff)

    def _check_updates(self):
        if not paths.is_frozen():
            messagebox.showinfo(
                "Check for Updates",
                "This button is for the packaged version. In development, use git pull.",
            )
            return

        token = os.environ.get("GITHUB_TOKEN", "").strip() or None

        try:
            remote_version = self._fetch_remote_version(token)
        except Exception as e:
            messagebox.showerror("Check for Updates", f"Could not read the remote version.\n\n{e}")
            return

        if remote_version == __version__:
            messagebox.showinfo("Check for Updates", f"You are up to date (v{__version__}).")
            return

        if not messagebox.askyesno(
            "Check for Updates",
            f"New version detected (local v{__version__} -> remote v{remote_version}).\n\n"
            "Download from GitHub Actions and install now? This will close TakoWorks and relaunch it.",
        ):
            return

        try:
            run_info = self._latest_actions_artifact(token)
        except Exception as e:
            messagebox.showerror(
                "Check for Updates",
                "Could not locate the latest build in GitHub Actions.\n\n"
                f"{e}\n\nMake sure the {WORKFLOW_FILE} workflow has completed successfully and use GITHUB_TOKEN.",
            )
            return

        try:
            payload_dir, temp_root = self._download_artifact(run_info, token)
        except Exception as e:
            messagebox.showerror("Check for Updates", f"Failed to download the artifact.\n\n{e}")
            return

        try:
            self._schedule_apply(payload_dir, temp_root, run_info, remote_version)
        except Exception as e:
            messagebox.showerror("Check for Updates", f"Failed to prepare the update.\n\n{e}")

    def _headers(self, token: str | None = None) -> dict:
        headers = {"User-Agent": USER_AGENT}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _fetch_remote_version(self, token: str | None) -> str:
        req = urllib.request.Request(RAW_VERSION_URL, headers=self._headers(token))
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status} while reading the remote version.")
            text = resp.read().decode("utf-8")
        match = re.search(r'__version__\\s*=\\s*"([^"]+)"', text)
        if not match:
            raise RuntimeError("Could not find __version__ in main.")
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
            raise RuntimeError("No successful workflow runs were found.")
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
            raise RuntimeError("No artifacts were found in the latest run.")
        artifact = next((a for a in artifacts if a.get("name") == ARTIFACT_NAME), artifacts[0])
        download_url = artifact.get("archive_download_url")
        if not download_url:
            raise RuntimeError("Could not find archive_download_url.")
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
                msg = "404 (you need GITHUB_TOKEN to download artifacts)"
            raise RuntimeError(msg)

        with zipfile.ZipFile(artifact_zip) as zf:
            zf.extractall(temp_root)

        inner_zip = None
        for candidate in temp_root.rglob("*.zip"):
            if candidate.name == ARTIFACT_ZIP:
                inner_zip = candidate
                break
        if not inner_zip:
            raise RuntimeError(f"Could not find {ARTIFACT_ZIP} inside the artifact.")

        payload_dir = temp_root / "payload"
        payload_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(inner_zip) as zf:
            zf.extractall(payload_dir)

        self.runner._console_write(
            f"[Update] Downloaded run #{run_info.get('run_number')} (id {run_info.get('run_id')}) to {payload_dir}"
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
            "Check for Updates",
            "Download complete. TakoWorks will close to apply the update and then relaunch.",
        )
        self.winfo_toplevel().after(300, self.winfo_toplevel().destroy)
