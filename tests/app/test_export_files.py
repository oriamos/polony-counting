import csv
import io
import json
import zipfile

from fakes import counting, exclusion, make_state, obj, rect
from polony_app.export import exporter
from polony_app.review import engine
from polony_app.schemas import ObjectClass, ReportingMode, SlideMeta


def rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def state():
    objs = [obj("r", 10, 10, ObjectClass.red), obj("g", 20, 10, ObjectClass.green), obj("y", 30, 10, ObjectClass.yellow),
            obj("u", 40, 10, ObjectClass.uncertain), obj("del", 50, 10), obj("ex", 110, 20)]
    st = make_state(objs, [counting(), exclusion()])
    engine.apply_action(st, "delete", {"object_ids": ["del"]})
    return st


def test_objects_csv_columns_and_content():
    a = state().analysis
    r = rows(exporter.objects_csv(a))
    assert [x["id"] for x in r] == ["r", "g", "y", "u", "ex"]  # deleted objects omitted
    for col in ["id", "x", "y", "class", "origin", "region_id", "red_detected", "red_snr", "red_peak", "red_saturated",
                "green_detected", "green_snr", "green_peak", "green_saturated", "association_distance_px",
                "review_reasons", "reviewed", "status", "counted"]:
        assert col in r[0]
    assert {x["status"] for x in r} == {"DRAFT"}
    byid = {x["id"]: x for x in r}
    assert byid["ex"]["region_id"] == "" and byid["ex"]["counted"] == "false"
    assert byid["y"]["red_detected"] == "true" and byid["y"]["green_detected"] == "true"
    assert byid["r"]["green_detected"] == "false" and byid["r"]["green_snr"] == ""


def test_regions_csv_both_conventions_unambiguous_and_uncertain_separate():
    a = state().analysis
    r = {x["region_id"]: x for x in rows(exporter.regions_csv(a))}
    c = r["c1"]
    assert (c["red_only"], c["green_only"], c["yellow"], c["uncertain"]) == ("1", "1", "1", "1")
    assert c["red_positive_incl_yellow"] == "2" and c["green_positive_incl_yellow"] == "2" and c["yellow_overlap"] == "1"
    assert c["total_objects_exclusive"] == "3"  # uncertain not included
    assert c["approval_status"] == "not_approved" and float(c["area_px2"]) > 0
    assert r["x1"]["kind"] == "exclusion" and r["x1"]["red_only"] == ""
    # area of c1 excludes the overlapping exclusion
    assert float(c["area_px2"]) < 150 * 110


def test_slide_csv_gates_concentration_and_never_hardcodes_volume():
    st = state()
    a = st.analysis
    a.slide.concentration_formula_confirmed = True
    s = rows(exporter.slide_csv(a))[0]
    assert s["concentration_status"].startswith("inputs incomplete: ")
    for m in ("sample_volume_ul", "concentration_factor"):
        assert m in s["concentration_status"]
    assert all(s[k] == "" for k in s if k.startswith("est_conc"))
    assert s["red_only"] == "1" and s["uncertain"] == "1" and s["sample_volume_ul"] == ""
    a.slide = SlideMeta(sample_volume_ul=2.5, concentration_factor=2.0)
    engine.update_counting_area(a)
    s = rows(exporter.slide_csv(a))[0]  # all inputs present but formula NOT confirmed
    assert s["concentration_status"] == "inputs incomplete: formula/geometry not confirmed by researcher"
    assert all(s[k] == "" for k in s if k.startswith("est_conc")) and s["concentration_formula"] == ""
    a.slide.concentration_formula_confirmed = True
    s = rows(exporter.slide_csv(a))[0]
    assert s["concentration_status"].startswith("computed") and "uncertain" in s["concentration_status"]
    assert float(s["est_conc_per_ml_red_only"]) == 1 / 0.0025 * 2.0
    assert float(s["est_conc_per_ml_total_exclusive"]) == 3 / 0.0025 * 2.0
    assert s["concentration_formula"]
    # same inputs but unconfirmed mapping => withheld
    st2 = make_state([obj("r", 10, 10)], [counting()], mapped=False)
    st2.analysis.slide = SlideMeta(sample_volume_ul=5, concentration_factor=1, concentration_formula_confirmed=True)
    s = rows(exporter.slide_csv(st2.analysis))[0]
    assert "confirmed_channel_mapping" in s["concentration_status"] and s["est_conc_per_ml_red_only"] == ""
    # missing counting region => area missing
    st3 = make_state([obj("r", 10, 10)], [])
    st3.analysis.slide = SlideMeta(sample_volume_ul=5, concentration_factor=1, concentration_formula_confirmed=True)
    assert "counting_area_px2" in rows(exporter.slide_csv(st3.analysis))[0]["concentration_status"]


def test_final_vs_draft_labels_and_no_confidence_interval():
    st = state()
    a = st.analysis
    assert exporter.export_filename(a, "objects").endswith("_DRAFT.csv")
    zdraft = zipfile.ZipFile(io.BytesIO(exporter.bundle_zip(st)))
    assert zdraft.read("README.txt").decode().startswith("DRAFT")
    assert "objects_DRAFT.csv" in zdraft.namelist()
    # no CI column anywhere
    blob = " ".join(zdraft.read(n).decode().lower() for n in zdraft.namelist() if n.endswith(".csv"))
    assert "ci_" not in blob and "confidence" not in blob and "interval" not in blob
    readme = zdraft.read("README.txt").decode()
    assert "No confidence interval is provided" in readme and "Poisson" in readme
    assert "UNCONFIRMED" in readme and "mean(fields)" in readme
    st.analysis.finalized_at = "2026-01-01T00:00:00Z"
    assert exporter.export_filename(a, "objects") == "t_objects.csv"
    assert rows(exporter.objects_csv(a))[0]["status"] == "FINAL"
    assert exporter.readme_text(a).startswith("FINAL")


def test_provenance_content():
    st = state()
    st.analysis.reporting_mode = ReportingMode.channel_inclusive
    p = json.loads(exporter.provenance_json(st))
    for k in ["schema_version", "app_version", "package_versions", "reporting_mode", "provenance", "image", "regions",
              "detection_parameters", "action_log_summary", "alignment_applied_to_green_px", "status"]:
        assert k in p
    assert p["reporting_mode"] == "channel_inclusive" and p["status"] == "DRAFT"
    assert p["image"]["sha256"] == {"a.tif": "0" * 64} and p["image"]["bit_depth"] == 16
    assert [c["role"] for c in p["image"]["channel_mapping"]] == ["red", "green"]
    assert p["image"]["channel_mapping"][0]["role_confirmed"] is True
    assert p["action_log_summary"]["by_type"] == {"delete": 1}
    assert "numpy" in p["package_versions"]
