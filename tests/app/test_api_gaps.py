"""Contract gap closures: per-channel preview, list counts, error messages, valid-region display range."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from fakes import counting, detector, exclusion, rect, write_two_channel_tiff  # noqa: F401
from polony_app.api.main import create_app
from test_api_flow import DETS, act, detect, upload  # noqa: F401

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "data")) as c:
        yield c


@pytest.fixture
def tiff(tmp_path):
    p = tmp_path / "s.tif"
    write_two_channel_tiff(p)
    return p


def test_channel_preview_before_mapping_and_swap(client, tiff):
    aid = upload(client, tiff)["id"]
    r0 = client.get(f"/api/analyses/{aid}/image/ch0", params={"max_dim": 64})
    r1 = client.get(f"/api/analyses/{aid}/image/ch1", params={"max_dim": 64})
    assert r0.status_code == 200 and r0.content[:4] == b"\x89PNG" and r1.status_code == 200
    assert "X-Auto-Lo" in r0.headers and r0.headers["X-Image-Width"]
    assert r0.content != r1.content
    assert client.get(f"/api/analyses/{aid}/image/ch7").status_code == 404
    assert client.get(f"/api/analyses/{aid}/image/chx").json()["code"] == "invalid_view"
    # role views still refuse before mapping
    assert client.get(f"/api/analyses/{aid}/image/red").status_code == 409
    # swapping the mapping swaps red/green views but not ch views
    m = lambda a, b: client.put(f"/api/analyses/{aid}/channels", json={"mapping": [{"index": 0, "role": a}, {"index": 1, "role": b}]})
    m("red", "green")
    red_a = client.get(f"/api/analyses/{aid}/image/red", params={"max_dim": 64}).content
    m("green", "red")
    red_b = client.get(f"/api/analyses/{aid}/image/red", params={"max_dim": 64}).content
    assert red_a != red_b
    assert client.get(f"/api/analyses/{aid}/image/ch0", params={"max_dim": 64}).content == r0.content
    assert client.get(f"/api/analyses/{aid}/image/green", params={"max_dim": 64}).content != b""


def test_channel_preview_composite_only(client, tmp_path):
    from synthetic import default_scene, write_rgb
    sc = default_scene()
    p = write_rgb(tmp_path / "rgb.tif", sc.red, sc.green)
    aid = upload(client, p)["id"]
    for i in range(3):
        r = client.get(f"/api/analyses/{aid}/image/ch{i}", params={"max_dim": 64})
        assert r.status_code == 200, r.text


def test_list_counts_and_status(client, tiff, detector):
    aid = upload(client, tiff)["id"]
    client.put(f"/api/analyses/{aid}/channels", json={"mapping": [{"index": 0, "role": "red"}, {"index": 1, "role": "green"}]})
    client.put(f"/api/analyses/{aid}/regions", json={"regions": [counting("c1", rect(5, 5, 155, 115)).model_dump(mode="json")]})
    d = detect(client, aid, detector, DETS())
    assert d["status"] == "in_review"
    item = client.get("/api/analyses").json()[0]
    assert item["counts"]["yellow"] == 1 and item["status"] == "in_review"


def test_refusal_messages_are_human_readable(client, tiff, detector):
    aid = upload(client, tiff)["id"]
    client.put(f"/api/analyses/{aid}/channels", json={"mapping": [{"index": 0, "role": "red"}, {"index": 1, "role": "green"}]})
    client.put(f"/api/analyses/{aid}/regions", json={"regions": [counting("c1", rect(5, 5, 155, 115)).model_dump(mode="json")]})
    detect(client, aid, detector, DETS())
    r = client.post(f"/api/analyses/{aid}/finalize")
    j = r.json()
    assert r.status_code == 409 and "not approved" in j["message"] and j["detail"]["problems"]
    r = act(client, aid, "approve_region", region_id="c1")
    assert r.status_code == 409 and "inspected" in r.json()["message"]
    act(client, aid, "region_inspected", region_id="c1", inspected=True)
    r = act(client, aid, "approve_region", region_id="c1")
    assert "uncertain" in r.json()["message"] and r.json()["detail"]["object_ids"]


def test_thumb_deleted_and_unknown(client, tiff, detector):
    aid = upload(client, tiff)["id"]
    client.put(f"/api/analyses/{aid}/channels", json={"mapping": [{"index": 0, "role": "red"}, {"index": 1, "role": "green"}]})
    client.put(f"/api/analyses/{aid}/regions", json={"regions": [counting("c1", rect(5, 5, 155, 115)).model_dump(mode="json")]})
    detect(client, aid, detector, DETS())
    act(client, aid, "delete", object_ids=["d1"])
    r = client.get(f"/api/analyses/{aid}/thumb/d1")
    assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
    r = client.get(f"/api/analyses/{aid}/thumb/nope")
    assert r.status_code == 404 and r.json()["code"] == "object_not_found" and r.json()["message"]


def test_display_range_uses_valid_regions(client, tmp_path):
    from synthetic import make_scene, write_two_channel
    sc = make_scene((200, 300), [(150, 100, 3000)], [], edge_band=60)  # saturated band outside the region
    p = write_two_channel(tmp_path / "e.tif", sc.red, sc.green)
    aid = upload(client, p)["id"]
    client.put(f"/api/analyses/{aid}/channels", json={"mapping": [{"index": 0, "role": "red"}, {"index": 1, "role": "green"}]})
    hi_all = float(client.get(f"/api/analyses/{aid}/image/red", params={"max_dim": 32}).headers["X-Auto-Hi"])
    client.put(f"/api/analyses/{aid}/regions", json={"regions": [counting("c1", rect(100, 10, 290, 190)).model_dump(mode="json")]})
    hi_reg = float(client.get(f"/api/analyses/{aid}/image/red", params={"max_dim": 32}).headers["X-Auto-Hi"])
    assert hi_all > 60000 and hi_reg < 8000
    # explicit lo/hi still honoured
    r = client.get(f"/api/analyses/{aid}/image/red", params={"lo": 0, "hi": 100, "max_dim": 32})
    assert r.status_code == 200
