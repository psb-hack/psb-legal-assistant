"""Прогон ZIP как на платформе: пустой каталог, фейковый прокси, лимит 20 мин."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from build_archive import ArchiveManifest, build
from fake_proxy import FakeOpenAIServer, ProxyScript

PLATFORM_TIMEOUT_S = 20 * 60


def extract_archive(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    return dest


def run_local(
    workdir: Path,
    data_dir: Path,
    *,
    timeout_s: int = PLATFORM_TIMEOUT_S,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["DATA_DIR"] = str(data_dir)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "solution.py"],
        cwd=workdir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def docker_available() -> bool:
    return shutil.which("docker") is not None


def run_docker(
    repo_root: Path,
    *,
    timeout_s: int = PLATFORM_TIMEOUT_S,
) -> subprocess.CompletedProcess[str]:
    if not docker_available():
        raise RuntimeError("docker не установлен")
    compose = repo_root / "docker" / "compose.yaml"
    env = os.environ.copy()
    env["PLATFORM_TIMEOUT"] = str(timeout_s)
    return subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(compose),
            "up",
            "--build",
            "--abort-on-container-exit",
            "--exit-code-from",
            "runner",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )


def pack_and_extract(repo_root: Path, dist: Path, run_dir: Path) -> Path:
    archive = dist / "submission.zip"
    build(ArchiveManifest.default(repo_root), archive)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    return extract_archive(archive, run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Прогон ZIP как на платформе")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--mode", choices=("local", "docker"), default="local")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=PLATFORM_TIMEOUT_S)
    args = parser.parse_args()
    data_dir = args.data if args.data is not None else args.root / "data"
    dist = args.root / "dist"
    run_dir = dist / "platform-run"
    pack_and_extract(args.root, dist, run_dir)
    if args.mode == "docker":
        result = run_docker(args.root, timeout_s=args.timeout)
    else:
        proxy = FakeOpenAIServer(script=ProxyScript())
        extra = {
            "LLM_BASE_URL": proxy.start(),
            "LLM_API_KEY": "local-fake",
            "LLM_MODEL": "deepseek",
            "EMB_MODEL": "bge-m3",
            "RERANKER_MODEL": "qwen-rerank",
        }
        try:
            result = run_local(
                run_dir, data_dir, timeout_s=args.timeout, extra_env=extra
            )
        finally:
            proxy.stop()
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)
    print(run_dir / "submission.csv")


if __name__ == "__main__":
    main()
