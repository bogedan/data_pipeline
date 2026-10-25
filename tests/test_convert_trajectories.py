from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "03_convert_trajectories.py"
SPEC = importlib.util.spec_from_file_location("convert_trajectories", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def config(source: Path, output: Path, *, overwrite: bool = False) -> dict:
    return {
        "_config_path": str(source.parent / "config.yaml"),
        "source": {"root": str(source)},
        "output": {"root": str(output), "overwrite": overwrite},
    }


def test_prepare_output_copies_all_source_assets(tmp_path: Path) -> None:
    source = tmp_path / "source_dataset"
    output = tmp_path / "output_dataset"
    calibration = source / "assets" / "calibration"
    calibration.mkdir(parents=True)
    (source / "assets" / "task.yaml").write_text("dataset:\n  name: demo\n", encoding="utf-8")
    (source / "assets" / "layout.rbl").write_text("blueprint", encoding="utf-8")
    (source / "assets" / "stale.rrd").write_bytes(b"old trajectory")
    (calibration / "camera.yaml").write_text("rms: 0.2\n", encoding="utf-8")

    result = MODULE.prepare_output(config(source, output), source)

    assert result == output.resolve()
    assert (output / "assets" / "task.yaml").read_text(encoding="utf-8") == "dataset:\n  name: demo\n"
    assert (output / "assets" / "layout.rbl").read_text(encoding="utf-8") == "blueprint"
    assert (output / "assets" / "calibration" / "camera.yaml").read_text(encoding="utf-8") == "rms: 0.2\n"
    assert not (output / "assets" / "stale.rrd").exists()
    marker = json.loads((output / MODULE.PIPELINE_MARKER).read_text(encoding="utf-8"))
    assert marker["complete"] is False


def test_missing_source_assets_does_not_remove_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source_dataset"
    output = tmp_path / "output_dataset"
    source.mkdir()
    output.mkdir()
    (output / MODULE.PIPELINE_MARKER).write_text("{}\n", encoding="utf-8")
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="assets directory does not exist"):
        MODULE.prepare_output(config(source, output, overwrite=True), source)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_prepare_output_keeps_preflight_inspection_report(tmp_path: Path) -> None:
    source = tmp_path / "source_dataset"
    output = tmp_path / "output_dataset"
    (source / "assets").mkdir(parents=True)
    report = output / "work" / "02_inspection.json"
    report.parent.mkdir(parents=True)
    report.write_text('{"source": "checked"}\n', encoding="utf-8")

    MODULE.prepare_output(config(source, output), source)

    assert json.loads(report.read_text(encoding="utf-8")) == {"source": "checked"}
    assert (output / MODULE.PIPELINE_MARKER).is_file()


def test_prepare_output_preserves_new_inspection_when_rebuilding(tmp_path: Path) -> None:
    source = tmp_path / "source_dataset"
    output = tmp_path / "output_dataset"
    (source / "assets").mkdir(parents=True)
    report = output / "work" / "02_inspection.json"
    report.parent.mkdir(parents=True)
    report.write_text('{"run": 2}\n', encoding="utf-8")
    (output / MODULE.PIPELINE_MARKER).write_text('{"complete": true}\n', encoding="utf-8")
    (output / "old.txt").write_text("old", encoding="utf-8")

    MODULE.prepare_output(config(source, output, overwrite=True), source)

    assert json.loads(report.read_text(encoding="utf-8")) == {"run": 2}
    assert not (output / "old.txt").exists()
