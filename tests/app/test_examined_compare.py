"""Comparison report only over fields the researcher provably examined (docs/app/FIELDS.md, fields/examined.py)."""
import json

import pytest
from fastapi.testclient import TestClient

from legacy_fixtures import GRID_XS, GRID_YS, blank, grid_overlay, point_roi, write_stack
from polony_app.api.main import create_app

INFO = "ImageDescription: 635 [W1]\nSoftware: ImageIO 2.0.4.0"
# counter 0 -> field 1 (cell 0,0): 3 marks on page 0 ; counter 1 -> field 2 (cell 0,1): 2 marks on page 0 ;
# counter 2 -> field 3 (cell 1,0): 1 mark on page 0 + 1 on page 1.  Fields 4.. (other cells) carry no marks.
PTS = [(20.5, 20.5), (25.5, 22.5), (30.5, 30.5), (70.5, 20.5), (80.5, 25.5), (20.5, 60.5), (22.5, 62.5)]
CNT = [0, 0, 0, 1, 1, 2, 2]
POS = [1, 1, 1, 1, 1, 1, 2]
RED = [(20, 20), (30, 30), (70, 20), (80, 25), (70, 60), (100, 60)]   # f1:2  f2:2  f5 (cell 1,1): 2 (unmarked field)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "data")) as c:
        yield c


@pytest.fixture
def aid(client, tmp_path):
    p = write_stack(tmp_path / "s.tif", [blank(), blank()], roi=point_roi(PTS, CNT, POS),
                    overlays=[grid_overlay(GRID_XS, GRID_YS, 160, 120)], info=INFO)
    with open(p, "rb") as f:
        r = client.post("/api/analyses", files=[("files", (p.name, f, "image/tiff"))])
    a = r.json()["id"]
    client.put(f"/api/analyses/{a}/channels", json={"mapping": [{"index": 0, "role": "red"}, {"index": 1, "role": "green"}]})
    assert client.post(f"/api/analyses/{a}/fields/import", json={"mode": "all"}).status_code == 200
    for x, y in RED:
        assert client.post(f"/api/analyses/{a}/actions", json={"type": "add", "payload": {"x": x, "y": y, "cls": "red"}}).status_code == 200
    return a


def manifest(*pages, evidence="T7B sheet, blank = not counted, 0 = counted zero", kind="sidecar_csv", sha=None):
    f = {"file": "s.tif", "pages": {str(pg): ents for pg, ents in pages}}
    if sha:
        f["sha256"] = sha
    return {"schema": "polony-examined-fields", "version": 1, "evidence": evidence, "source_kind": kind, "files": [f]}


def upload(client, aid, doc, name="m.json"):
    body = doc if isinstance(doc, (bytes, str)) else json.dumps(doc)
    return client.post(f"/api/analyses/{aid}/examined-fields", files={"file": (name, body, "application/json")})


def cmp(client, aid, plane=0, ct="red_positive"):
    r = client.get(f"/api/analyses/{aid}/fields/compare?plane={plane}&count_type={ct}")
    assert r.status_code == 200, r.text
    return r.json()


def by_field(j):
    return {r["field"]: r for r in j["rows"]}


def test_page0_marks_make_only_page0_field_eligible(client, aid):
    j0, j1 = cmp(client, aid, 0), cmp(client, aid, 1)
    s0, s1 = j0["summary"], j1["summary"]
    assert s0["n_fields"] == 9 and s0["eligible_examined_fields"] == 3 and s0["positive_marked_fields"] == 3
    assert s0["unknown_fields_excluded"] == 6 and s0["verified_zero_fields"] == 0 and s0["conflict_fields_excluded"] == 0
    assert {f for f, r in by_field(j0).items() if r["examined"]} == {1, 2, 3}
    # the same fields on page 1: only field 3 has a mark there; fields 1 and 2 stay unknown (never zero)
    assert {f for f, r in by_field(j1).items() if r["examined"]} == {3}
    assert by_field(j1)[1]["status"] == "unknown" and by_field(j1)[1]["reference_count"] is None
    assert by_field(j1)[1]["delta"] is None and by_field(j1)[1]["examined"] is False
    assert s1["eligible_examined_fields"] == 1 and s1["total_reference"] == 1


def test_unmarked_field_is_excluded_from_every_metric_and_its_app_objects_never_enter_totals(client, aid):
    j = cmp(client, aid, 0)
    f5 = by_field(j)[5]
    assert f5["app_count"] == 2 and f5["status"] == "unknown" and f5["reference_count"] is None   # shown ...
    s = j["summary"]
    assert (s["total_reference"], s["total_app"]) == (6, 4)          # ... but not in the totals (f1 2 + f2 2 + f3 0)
    assert s["app_total_unknown_fields_not_counted"] == 2
    assert s["mae"] == pytest.approx((1 + 0 + 1) / 3)
    assert s["slope"] == pytest.approx((3 * 2 + 2 * 2 + 0) / (9 + 4 + 1))
    assert s["eligibility_source"] == "embedded_imagej_marks"
    assert not any("compared as 0" in w for w in j["warnings"]) and any("never read as zero" in w for w in j["warnings"])


def test_unknown_fields_do_not_change_metrics(client, aid):
    def metrics(j):
        s = j["summary"]
        return {k: s[k] for k in ("pearson_r", "slope", "mae", "total_reference", "total_app", "eligible_examined_fields")}
    before = metrics(cmp(client, aid, 0))
    # add many app objects into other unknown fields (cell 0,2 = field 4 and cell 2,2)
    for x, y in [(120, 20), (130, 25), (140, 30), (120, 100), (130, 100)]:
        client.post(f"/api/analyses/{aid}/actions", json={"type": "add", "payload": {"x": x, "y": y, "cls": "red"}})
    j = cmp(client, aid, 0)
    assert by_field(j)[4]["app_count"] == 3 and by_field(j)[4]["status"] == "unknown"
    assert metrics(j) == before


def test_manifest_confirmed_zero_is_included_with_reference_zero(client, aid):
    r = upload(client, aid, manifest((0, [{"field": 5, "count": 0}])))
    assert r.status_code == 200, r.text
    assert r.json()["matched_file_entries"] == 1 and r.json()["n_entries"] == 1 and r.json()["pages"] == {"0": 1}
    j = cmp(client, aid, 0)
    f5 = by_field(j)[5]
    assert (f5["status"], f5["examined"], f5["reference_count"], f5["delta"]) == ("verified_zero", True, 0, 2)
    assert f5["examination_source"] == "sidecar_csv"
    s = j["summary"]
    assert s["verified_zero_fields"] == 1 and s["eligible_examined_fields"] == 4 and s["unknown_fields_excluded"] == 5
    assert (s["total_reference"], s["total_app"]) == (6, 6)
    assert s["eligibility_source"] == "embedded_imagej_marks+sidecar_csv"
    # evidence is per plane: field 5 is still unknown on page 1
    assert by_field(cmp(client, aid, 1))[5]["status"] == "unknown"


def test_manifest_entry_without_count_and_without_marks_stays_unknown(client, aid):
    assert upload(client, aid, manifest((0, [{"field": 5}]))).status_code == 200
    f5 = by_field(cmp(client, aid, 0))[5]
    assert f5["status"] == "unknown" and f5["reference_count"] is None and "zero is not proven" in f5["reason"]


def test_manifest_count_contradicting_marks_is_conflict_and_excluded(client, aid):
    assert upload(client, aid, manifest((0, [{"field": 1, "count": 5}, {"counter": 1, "count": 0}]))).status_code == 200
    j = cmp(client, aid, 0)
    r = by_field(j)
    assert r[1]["status"] == "conflict" and r[1]["reference_count"] is None and r[1]["examined"] is False
    assert r[2]["status"] == "conflict"                      # counter 1 = field 2: stated zero but 2 marks
    s = j["summary"]
    assert s["conflict_fields_excluded"] == 2 and s["eligible_examined_fields"] == 1
    assert (s["total_reference"], s["total_app"]) == (1, 0)


def test_manifest_resolves_counter_and_cell_and_warns_on_unresolved(client, aid):
    doc = manifest((0, [{"counter": 4, "count": 0}, {"row": 2, "col": 2, "count": 0}, {"field": 77, "count": 0}]))
    r = upload(client, aid, doc)
    assert r.status_code == 200
    j = cmp(client, aid, 0)
    rows = by_field(j)
    assert rows[5]["status"] == "verified_zero"             # counter 4 = field 5 (index numbering)
    assert rows[9]["status"] == "verified_zero"             # grid cell row 2 col 2
    assert any("field 77" in w and "ignored" in w for w in j["warnings"])


def test_no_eligible_fields_gives_null_metrics_and_warning(client, aid):
    j = cmp(client, aid, 5)
    s = j["summary"]
    assert s["eligible_examined_fields"] == 0 and s["eligibility_source"] == "none"
    for k in ("pearson_r", "slope", "mae", "total_reference", "total_app"):
        assert s[k] is None
    assert s["unknown_fields_excluded"] == 9
    assert any("No field was provably examined" in w for w in j["warnings"])


def test_examined_endpoint_validates_stores_and_respects_finalized(client, aid):
    for bad in ("not json", json.dumps({"schema": "x"}), json.dumps(manifest((0, []), evidence=" "))):
        r = upload(client, aid, bad)
        assert r.status_code == 422 and r.json()["code"] == "examined_manifest_invalid"
    assert upload(client, aid, manifest((0, [{"field": 5, "count": 0, "row": 1, "col": 1}]))).status_code == 422
    # nothing stored by the failures: field 5 still unknown
    assert by_field(cmp(client, aid, 0))[5]["status"] == "unknown"
    csv = "# evidence: T7B sheet, 0 = counted zero\nfile,page,field,count\ns.tif,0,5,0\ns.tif,0,6,\n"
    r = upload(client, aid, csv, "m.csv")
    assert r.status_code == 200 and r.json()["n_entries"] == 2 and r.json()["stored_as"] == "examined_fields.json"
    rows = by_field(cmp(client, aid, 0))
    assert rows[5]["status"] == "verified_zero" and rows[6]["status"] == "unknown"
    # a manifest for another file matches nothing: accepted, with a warning, and changes nothing
    other = manifest((0, [{"field": 7, "count": 0}]))
    other["files"][0]["file"] = "other.tif"
    r = upload(client, aid, other)
    assert r.status_code == 200 and r.json()["matched_file_entries"] == 0 and r.json()["warnings"]
    assert by_field(cmp(client, aid, 0))[5]["status"] == "unknown"      # replaced
    with client.app.state.store.edit(aid) as st:           # finalized analyses are read-only for every mutation
        st.analysis.finalized_at = "2026-10-01T10:00:00Z"
    r = upload(client, aid, manifest((0, [{"field": 5, "count": 0}])))
    assert r.status_code == 409 and r.json()["code"] == "analysis_finalized"


def test_manifest_with_wrong_sha256_is_refused(client, aid):
    r = upload(client, aid, manifest((0, [{"field": 5, "count": 0}]), sha="0" * 64))
    assert r.status_code == 422 and r.json()["code"] == "examined_manifest_mismatch"
