"""Originals stay local: original_missing / attach flow (both backends), two computers on one database."""
import hashlib
import os
import shutil
import uuid

import pytest
from fastapi.testclient import TestClient

from fakes import write_two_channel_tiff
from polony_app.api.main import create_app
from polony_app.config import Settings

URL = os.environ.get("POLONY_TEST_DATABASE_URL", "")


def _upload(c, tiff, name="slide A"):
    with open(tiff, "rb") as f:
        r = c.post("/api/analyses", files=[("files", (tiff.name, f, "image/tiff"))], data={"name": name})
    assert r.status_code == 200, r.text
    aid = r.json()["id"]
    r2 = c.put(f"/api/analyses/{aid}/channels", json={"mapping": [{"index": 0, "role": "red"}, {"index": 1, "role": "green"}]})
    assert r2.status_code == 200, r2.text
    return r2.json()


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def client(data_dir):
    with TestClient(create_app(data_dir)) as c:
        yield c


@pytest.fixture
def tiff(tmp_path):
    p = tmp_path / "scan one.tif"
    write_two_channel_tiff(p)
    return p


def _attach(c, aid, path, filename=None):
    with open(path, "rb") as f:
        return c.post(f"/api/analyses/{aid}/originals", files=[("files", (filename or path.name, f, "image/tiff"))])


def test_original_missing_then_attach_and_sha_mismatch(client, data_dir, tiff, tmp_path, backend):
    a = _upload(client, tiff)
    aid = a["id"]
    assert client.get(f"/api/analyses/{aid}/image/composite").status_code == 200
    expected = a["image"]["sha256"]
    # simulate "analysis created on another computer": the local originals are gone
    shutil.rmtree(data_dir / "originals" / aid)
    for url in (f"/api/analyses/{aid}/image/composite", f"/api/analyses/{aid}/histogram/red"):
        r = client.get(url)
        assert r.status_code == 409 and r.json()["code"] == "original_missing", r.text
        files = r.json()["detail"]["files"]
        assert {f["file"]: f["sha256"] for f in files} == expected
    # analysis data and table exports still work without pixels
    assert client.get(f"/api/analyses/{aid}").status_code == 200
    assert client.get(f"/api/analyses/{aid}/export", params={"kind": "regions"}).status_code == 200
    # wrong file: refused, nothing stored
    bad = tmp_path / "other.tif"
    write_two_channel_tiff(bad, seed=7)
    r = _attach(client, aid, bad)
    assert r.status_code == 422 and r.json()["code"] == "sha256_mismatch"
    assert client.get(f"/api/analyses/{aid}/image/composite").status_code == 409
    assert not list((data_dir / "originals" / aid).glob("*")) if (data_dir / "originals" / aid).exists() else True
    # right file under another name: accepted, stored under the recorded name
    renamed = tmp_path / "my copy (1).tif"
    shutil.copy(tiff, renamed)
    r = _attach(client, aid, renamed)
    assert r.status_code == 200 and r.json()["missing"] == []
    assert hashlib.sha256((data_dir / "originals" / aid / list(expected)[0]).read_bytes()).hexdigest() == list(expected.values())[0]
    assert client.get(f"/api/analyses/{aid}/image/composite").status_code == 200


@pytest.mark.parametrize("backend", ["postgres"], indirect=True)
def test_two_computers_share_one_database(tmp_path, tiff, backend, monkeypatch):
    monkeypatch.setenv("POLONY_PG_VIEW_TTL", "0")
    schema = os.environ["POLONY_DATABASE_SCHEMA"]
    sa = Settings(database_url=URL, database_schema=schema, machine_id=uuid.uuid4().hex)
    sb = Settings(database_url=URL, database_schema=schema, machine_id=uuid.uuid4().hex)
    with TestClient(create_app(tmp_path / "A", sa)) as ca, TestClient(create_app(tmp_path / "B", sb)) as cb:
        a = _upload(ca, tiff)
        aid = a["id"]
        assert [x["id"] for x in cb.get("/api/analyses").json()] == [aid]
        r = cb.get(f"/api/analyses/{aid}/image/composite")                 # B has no TIFF
        assert r.status_code == 409 and r.json()["code"] == "original_missing"
        assert ca.get("/api/storage").json()["machine_id"] != cb.get("/api/storage").json()["machine_id"]
        assert _attach(cb, aid, tiff).status_code == 200
        r = cb.get(f"/api/analyses/{aid}/image/composite")
        assert r.status_code == 200, r.text
        # an edit on A is the state B reads next, and vice versa
        r = ca.put(f"/api/analyses/{aid}/regions", json={"regions": [{
            "id": "c1", "kind": "counting", "name": "c1", "polygon": [[5, 5], [155, 5], [155, 115], [5, 115]]}]})
        assert r.status_code == 200, r.text
        got = cb.get(f"/api/analyses/{aid}").json()
        assert [x["id"] for x in got["regions"]] == ["c1"]
        assert cb.delete(f"/api/analyses/{aid}").status_code in (200, 204)
        assert ca.get(f"/api/analyses/{aid}").status_code == 404
