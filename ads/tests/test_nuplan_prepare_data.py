"""HTTP range and atomic extraction checks without downloading nuPlan logs."""

import io
import json
import zipfile

import pytest
import yaml

from ads.nuplan import prepare_data


@pytest.mark.parametrize("budget_gib", [4, 8])
def test_source_pool_respects_budget_and_separates_sessions(
    tmp_path, monkeypatch, budget_gib
):
    monkeypatch.setattr(prepare_data, "WORKSPACE", tmp_path)
    manifest = tmp_path / "preparation/plan.json"
    monkeypatch.setattr(prepare_data, "MANIFEST", manifest)
    parts = ("boston", "pittsburgh", "singapore", "vegas_1")
    monkeypatch.setattr(prepare_data, "ARCHIVES", parts)
    rows = {}
    for part in parts:
        city = "las_vegas" if part == "vegas_1" else part
        rows[part] = [
            {
                "archive": part,
                "city": city,
                "session": f"{city}_{index}_vehicle",
                "log": f"{city}_{index}_vehicle_000_100",
                "bytes": (160 if index == 1 else 32) * 1024**2,
                "compressed_bytes": 8 * 1024**2,
            }
            for index in range(20)
        ]
    splitter = tmp_path / (
        "nuplan-devkit/nuplan/planning/script/config/common/splitter/nuplan.yaml"
    )
    splitter.parent.mkdir(parents=True)
    splitter.write_text(
        yaml.safe_dump(
            {
                "log_splits": {
                    "train": [r["log"] for values in rows.values() for r in values],
                    "val": ["boston_0_vehicle_100_200"],
                    "test": [],
                }
            }
        )
    )

    def read_archive(part):
        return {"part": part, "city": rows[part][0]["city"]}, rows[part]

    monkeypatch.setattr(prepare_data, "read_archive", read_archive)
    plan = prepare_data.make_plan(budget_gib)
    selected = plan["logs"]
    assert plan["raw_bytes"] <= plan["raw_budget_bytes"]
    assert plan["raw_budget_bytes"] + plan["cache_and_temporary_reserve_bytes"] == (
        budget_gib * 1024**3
    )
    assert len({r["session"] for r in selected}) == len(selected)
    assert all(r["session"] != "boston_0_vehicle" for r in selected)
    for city in ("boston", "pittsburgh", "singapore", "las_vegas"):
        assert {r["purpose"] for r in selected if r["city"] == city} == {
            "replay",
            "development",
        }
    if budget_gib == 8:
        assert any(r["bytes"] > 96 * 1024**2 for r in selected)


def remote_zip(monkeypatch, *, corrupt=False, ignore_range=False):
    payload = b"selected driving log" * 400
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("log.db", payload)
        archive.writestr("unselected.db", b"x" * (2 * 1024**2))
    data = memory.getvalue()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        info = archive.getinfo("log.db")
    if corrupt:
        location = data.index(payload)
        data = data[:location] + bytes([data[location] ^ 1]) + data[location + 1 :]
    calls = []

    class Response(io.BytesIO):
        def __init__(self, body, status, headers):
            super().__init__(body)
            self.status, self.headers = status, headers

    def urlopen(request, timeout):
        if request.get_method() == "HEAD":
            return Response(b"", 200, {"Content-Length": str(len(data)), "ETag": "zip"})
        value = request.get_header("Range")
        assert value is not None
        start, end = map(int, value.removeprefix("bytes=").split("-"))
        calls.append((start, end))
        return Response(
            data[start : end + 1],
            200 if ignore_range else 206,
            {"Content-Range": f"bytes {start}-{end}/{len(data)}"},
        )

    monkeypatch.setattr(prepare_data.urllib.request, "urlopen", urlopen)
    return payload, data, info, calls


def test_selected_member_uses_ranges_and_never_fetches_whole_archive(monkeypatch):
    payload, data, _, calls = remote_zip(monkeypatch)
    with (
        prepare_data.RangeFile("https://example.org/train.zip") as source,
        zipfile.ZipFile(source) as archive,
    ):
        assert archive.read("log.db") == payload
    assert sum(end - start + 1 for start, end in calls) < len(data) // 4


def test_server_ignoring_ranges_is_rejected(monkeypatch):
    remote_zip(monkeypatch, ignore_range=True)
    with (
        prepare_data.RangeFile("https://example.org/train.zip") as source,
        pytest.raises(ValueError, match="requested byte range"),
    ):
        zipfile.ZipFile(source)


@pytest.mark.parametrize("corrupt", [False, True])
def test_download_publishes_only_complete_verified_logs(tmp_path, monkeypatch, corrupt):
    payload, data, info, _ = remote_zip(monkeypatch, corrupt=corrupt)
    monkeypatch.setattr(prepare_data, "WORKSPACE", tmp_path)
    manifest = tmp_path / "subset.json"
    monkeypatch.setattr(prepare_data, "MANIFEST", manifest)
    (tmp_path / "data/nuplan-v1.1/splits").mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "archives": [
                    {
                        "url": "https://example.org/train.zip",
                        "bytes": len(data),
                        "etag": "zip",
                        "city": "boston",
                        "part": "boston",
                    }
                ],
                "logs": [
                    {
                        "city": "boston",
                        "purpose": "replay",
                        "archive": "boston",
                        "log": "log",
                        "member": "log.db",
                        "bytes": info.file_size,
                        "compressed_bytes": info.compress_size,
                        "crc32": info.CRC,
                    }
                ],
            }
        )
    )
    target = tmp_path / "data/nuplan-v1.1/splits/train_replay/log.db"
    if corrupt:
        with pytest.raises(zipfile.BadZipFile, match="CRC"):
            prepare_data.download()
        assert not target.exists()
    else:
        prepare_data.download()
        assert target.read_bytes() == payload
        stamp = target.stat().st_mtime_ns
        prepare_data.download()
        assert target.stat().st_mtime_ns == stamp
