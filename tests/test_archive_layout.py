from __future__ import annotations

import csv
import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_archive_is_flat_and_runs(tmp_path: Path) -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        from build_archive import SRC_MODULES, ArchiveManifest, build
    finally:
        sys.path.pop(0)

    archive_path = tmp_path / "submission.zip"
    build(ArchiveManifest.default(REPO), archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
    assert set(names) == set(SRC_MODULES) | {"requirements.txt"}
    assert names == sorted(names)
    for name in names:
        assert "/" not in name
        assert "\\" not in name

    extract = tmp_path / "run"
    extract.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract)

    data_dir = extract / "data"
    data_dir.mkdir()
    (data_dir / "questions.csv").write_text(
        "id,question\nq1,Какой срок оплаты?\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["DATA_DIR"] = str(data_dir)
    result = subprocess.run(
        [sys.executable, "solution.py"],
        cwd=extract,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    submission = extract / "submission.csv"
    with submission.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["id"] for row in rows] == ["q1"]
    assert rows[0]["answer"].strip()
