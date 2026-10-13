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
    doc = manifest((0, [{"counter": 1, "count": 2}, {"field": 5, "count": 0}, {"row": 2, "col": 2, "count": 0}, {"field": 77, "count": 0}]))
    r = upload(client, aid, doc)
    assert r.status_code == 200
    j = cmp(client, aid, 0)
    rows = by_field(j)
    assert rows[2]["status"] == "positive_marked" and rows[2]["reference_count"] == 2   # counter 1 has points -> field 2
    assert rows[5]["status"] == "verified_zero"             # 'field' = the lab number the manifest author stated
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


# ---------------------------------------------------------------- F6: a counter without points is never guessed
def test_manifest_counter_without_points_is_not_guessed_as_counter_plus_one(client, aid):
    r = upload(client, aid, manifest((0, [{"counter": 5, "count": 0}, {"counter": 4, "count": 0}])))
    assert r.status_code == 200
    j = cmp(client, aid, 0)
    rows = by_field(j)
    assert rows[6]["status"] == "unknown" and rows[5]["status"] == "unknown"      # NOT verified_zero
    assert j["summary"]["verified_zero_fields"] == 0
    ws = [w for w in j["warnings"] if "counter 5 has no point" in w]
    assert ws and "counter+1" in ws[0] and "page 0: counter=5" in ws[0]
    assert any("counter 4 has no point" in w for w in j["warnings"])


# ---------------------------------------------------------------- F5: marks without a plane are evidence for NO plane
def _planeless_client(client, tmp_path, name, pos, cnt=CNT):
    p = write_stack(tmp_path / f"{name}.tif", [blank(), blank()], roi=point_roi(PTS, cnt, pos),
                    overlays=[grid_overlay(GRID_XS, GRID_YS, 160, 120)], info=INFO)
    with open(p, "rb") as f:
        a = client.post("/api/analyses", files=[("files", (p.name, f, "image/tiff"))]).json()["id"]
    client.put(f"/api/analyses/{a}/channels", json={"mapping": [{"index": 0, "role": "red"}, {"index": 1, "role": "green"}]})
    assert client.post(f"/api/analyses/{a}/fields/import", json={"mode": "all"}).status_code == 200
    return a


@pytest.mark.parametrize("pos", [None, [0] * 7, [3] * 7])
def test_marks_without_a_valid_plane_are_conflict_on_every_plane_and_never_counted(client, tmp_path, pos):
    a = _planeless_client(client, tmp_path, "np", pos)
    ref = client.get(f"/api/analyses/{a}/reference").json()
    assert all(p["plane"] is None for p in ref["points"]) and ref["planes"] == []
    assert ref["unplaned"]["count"] == 7 and ref["unplaned"]["fields"] == [1, 2, 3]
    assert ref["reference_counts"]["1"]["plane0"] == 0 and ref["reference_counts"]["1"]["unplaned"] == 3
    assert any("no valid plane" in w for w in ref["warnings"])
    for plane in (0, 1):
        j = cmp(client, a, plane)
        s = j["summary"]
        assert s["eligible_examined_fields"] == 0 and s["total_reference"] is None and s["mae"] is None
        assert s["unplaned_marks"] == 7 and s["unplaned_mark_fields"] == [1, 2, 3]
        rows = by_field(j)
        for f in (1, 2, 3):
            assert rows[f]["status"] == "conflict" and rows[f]["reference_count"] is None and rows[f]["examined"] is False
            assert rows[f]["reason"] == "mark with unspecified plane inside this field"
        assert j["warnings"][0].startswith("WARNING: 7 reference mark(s) have no valid plane")
        assert "1, 2, 3" in j["warnings"][0] and "ALL planes" in j["warnings"][0]


def test_one_planeless_mark_poisons_only_its_field_even_against_a_manifest(client, tmp_path):
    pos = [1, 1, 1, 1, 1, 1, 2]
    pos[0] = 0                                           # one mark of field 1 has no plane
    a = _planeless_client(client, tmp_path, "one", pos)
    assert upload(client, a, manifest((0, [{"field": 1, "count": 2}, {"field": 2, "count": 2}])), "m.json").status_code == 200
    for plane in (0, 1):
        j = cmp(client, a, plane)
        rows = by_field(j)
        assert rows[1]["status"] == "conflict" and "unspecified plane" in rows[1]["reason"]
        assert j["summary"]["unplaned_marks"] == 1 and j["summary"]["unplaned_mark_fields"] == [1]
    j0 = cmp(client, a, 0)
    assert by_field(j0)[2]["status"] == "positive_marked" and by_field(j0)[2]["reference_count"] == 2
    assert by_field(j0)[3]["status"] == "positive_marked" and j0["summary"]["total_reference"] == 3   # f2:2 + f3:1


def test_reference_counts_ignore_planeless_marks_unit():
    from polony_app.fields.reference import RefPoint, Reference
    r = Reference(True, points=[RefPoint(1, 1, 0, 0, 1, 1), RefPoint(2, 2, 0, None, 1, 1), RefPoint(3, 3, 1, 1, 2, 2),
                                RefPoint(4, 4, None, None, None, None)])
    assert r.counts("index", 0) == {1: 1} and r.counts("index", 1) == {2: 1}
    assert r.unplaned_fields() == [1] and len(r.unplaned_points()) == 2 and [p["page"] for p in r.planes()] == [0, 1]


# ---------------------------------------------------------------- shared model: strictness
def _write(tmp_path, text, name="m.json"):
    p = tmp_path / name
    p.write_bytes(text if isinstance(text, bytes) else text.encode())
    return p


def _doc(**entry):
    return {"schema": "polony-examined-fields", "version": 1, "evidence": "x",
            "files": [{"file": "s.tif", "pages": {"0": [entry]}}]}


@pytest.mark.parametrize("entry", [
    {"field": 1, "count": True}, {"field": 1, "count": 1.5}, {"field": 1, "count": "3"}, {"field": 1, "count": -1},
    {"field": 1, "count": 10 ** 6 + 1}, {"field": True, "count": 1}, {"field": -2, "count": 1}, {"counter": 1.0, "count": 1},
    {"row": -1, "col": 0, "count": 1}, {"row": "1", "col": 0, "count": 1}, {"field": 1, "count": 1, "cnt": 3},
    {"field": 1, "counted_zero": "yes"}, {"field": 1, "note": 5},
])
def test_manifest_entry_strictness(tmp_path, entry):
    from polony_app.fields import examined
    with pytest.raises(examined.ExaminedError):
        examined.load_manifest(_write(tmp_path, json.dumps(_doc(**entry))))


def test_manifest_accepts_plain_valid_entries_and_boundary_counts(tmp_path):
    from polony_app.fields import examined
    d = _doc(field=1, count=10 ** 6)
    d["files"][0]["pages"]["0"].append({"counter": 0, "counted_zero": True, "note": "ok"})
    m = examined.load_manifest(_write(tmp_path, json.dumps(d)))
    assert m.files[0].pages["0"][0].stated_count == 10 ** 6 and m.sha256


def test_manifest_unknown_keys_rejected_at_every_level(tmp_path):
    from polony_app.fields import examined
    top = _doc(field=1, count=1); top["extra"] = 1
    fil = _doc(field=1, count=1); fil["files"][0]["sha"] = "x"
    for d in (top, fil):
        with pytest.raises(examined.ExaminedError):
            examined.load_manifest(_write(tmp_path, json.dumps(d)))
    with pytest.raises(examined.ExaminedError, match="unknown CSV column"):
        examined.load_manifest(_write(tmp_path, "# evidence: x\nfile,page,field,count,cnt\ns.tif,0,1,0,1\n", "m.csv"))
    with pytest.raises(examined.ExaminedError, match="counted_zero"):
        examined.load_manifest(_write(tmp_path, "# evidence: x\nfile,page,field,counted_zero\ns.tif,0,1,Y\n", "m.csv"))


def test_manifest_deep_nesting_huge_and_binary_become_examined_error(tmp_path):
    from polony_app.fields import examined
    deep = "[" * 200000 + "]" * 200000
    for payload in (deep, '{"a":' * 50000 + "1" + "}" * 50000, b"\xff\xfe\x00garbage"):
        with pytest.raises(examined.ExaminedError):
            examined.load_manifest(_write(tmp_path, payload))
    big = _write(tmp_path, b" " * (examined.MAX_MANIFEST_FILE_BYTES + 1), "big.json")
    with pytest.raises(examined.ExaminedError, match="larger than"):
        examined.load_manifest(big)
    with pytest.raises(examined.ExaminedError):
        examined.load_manifest(tmp_path / "missing.json")


def test_entries_two_file_entries_same_file_different_content_is_an_error_not_last_wins():
    from polony_app.fields import examined as ex
    def man(c2):
        return ex.ExaminedManifest(evidence="x", files=[
            ex.FileEntry(file="a.tif", pages={"0": [ex.FieldEntry(field=1, count=5)]}),
            ex.FileEntry(file="x/a.tif", pages={"0": [ex.FieldEntry(field=1, count=c2)]})])
    with pytest.raises(ex.ExaminedError, match="silently win"):
        man(0).entries(["a.tif"])
    assert man(5).entries(["a.tif"])[0][("field", 1)].count == 5          # identical content is harmless


# ---------------------------------------------------------------- endpoint
def test_endpoint_rejects_deep_nesting_unknown_keys_huge_and_conflicting_entries(client, aid):
    r = upload(client, aid, "[" * 100000 + "]" * 100000)
    assert r.status_code == 422 and r.json()["code"] == "examined_manifest_invalid"
    bad = manifest((0, [{"field": 5, "count": 0, "cnt": 1}]))
    assert upload(client, aid, bad).status_code == 422
    bad = manifest((0, [{"field": 5, "count": True}]))
    assert upload(client, aid, bad).status_code == 422
    r = upload(client, aid, b" " * (2 * 1024 * 1024 + 1))
    assert r.status_code == 422 and "2 MB" in r.json()["message"]
    twice = manifest((0, [{"field": 5, "count": 0}]))
    twice["files"].append({"file": "x/s.tif", "pages": {"0": [{"field": 5, "count": 3}]}})
    assert upload(client, aid, twice).status_code == 422
    assert by_field(cmp(client, aid, 0))[5]["status"] == "unknown"       # nothing was stored by any failure
    # accepted uploads go to the fixed file name only, whatever the upload is called
    ok = upload(client, aid, manifest((0, [{"field": 5, "count": 0}])), "../../evil.json")
    assert ok.status_code == 200 and ok.json()["stored_as"] == "examined_fields.json"
    od = client.app.state.store.originals_dir(aid)
    assert (od / "examined_fields.json").is_file() and not (od.parent.parent / "evil.json").exists()
    assert by_field(cmp(client, aid, 0))[5]["status"] == "verified_zero"


def test_template_style_comment_key_is_allowed_but_carries_no_evidence(tmp_path):
    from polony_app.fields import examined
    d = _doc(field=1, count=1)
    d["_comment"] = ["fill me in"]
    assert examined.load_manifest(_write(tmp_path, json.dumps(d))).comment == ["fill me in"]
