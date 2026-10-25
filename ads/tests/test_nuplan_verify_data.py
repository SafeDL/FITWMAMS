"""Read-only database checks catch corrupt files and wrong map identifiers."""

import sqlite3
import zlib

import pytest

from ads.nuplan import verify_data


def make_database(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_data, "WORKSPACE", tmp_path)
    row = {"log": "example", "city": "las_vegas", "purpose": "replay"}
    path = verify_data.database_path(row)
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE log(location TEXT, map_version TEXT);
            INSERT INTO log VALUES('las_vegas','us-nv-las-vegas-strip');
            CREATE TABLE lidar_pc(token BLOB, timestamp INTEGER);
            CREATE TABLE ego_pose(token BLOB);
            INSERT INTO ego_pose VALUES(X'01');
            CREATE TABLE scenario_tag(type TEXT);
            INSERT INTO scenario_tag VALUES('following_lane_without_lead');
            CREATE TABLE scene(token BLOB);
            CREATE TABLE lidar_box(token BLOB);
            CREATE TABLE traffic_light_status(token BLOB);
            """
        )
        connection.executemany(
            "INSERT INTO lidar_pc VALUES(?,?)",
            [(bytes([index]), index * 5_000_000) for index in range(5)],
        )
    row.update(bytes=path.stat().st_size, crc32=zlib.crc32(path.read_bytes()))
    return row, path


def test_sqlite_readback_uses_map_version_not_generic_location(tmp_path, monkeypatch):
    row, path = make_database(tmp_path, monkeypatch)
    stamp = path.stat().st_mtime_ns
    record = verify_data.verify_database(row)
    assert record["map_name"] == "us-nv-las-vegas-strip"
    assert record["frames"] == 5
    assert record["duration_s"] == 20
    assert record["probe"] is not None
    assert len(record["sha256"]) == 64
    assert path.stat().st_mtime_ns == stamp


def test_same_size_corruption_is_rejected_before_sqlite_read(tmp_path, monkeypatch):
    row, path = make_database(tmp_path, monkeypatch)
    contents = path.read_bytes()
    path.write_bytes(contents[:-1] + bytes([contents[-1] ^ 1]))
    with pytest.raises(ValueError, match="CRC"):
        verify_data.verify_database(row)


def test_verified_crc_does_not_replace_required_table_check(tmp_path, monkeypatch):
    row, path = make_database(tmp_path, monkeypatch)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE ego_pose")
    row.update(bytes=path.stat().st_size, crc32=zlib.crc32(path.read_bytes()))
    with pytest.raises(ValueError, match="Missing nuPlan tables"):
        verify_data.verify_database(row)
