import csv
import io
import json
import zipfile

import pytest

from fakes import counting, exclusion, make_state, obj, rect
from polony_app.export import exporter
from polony_app.review import engine
from polony_app.schemas import LabCountType, LabSeries, ObjectClass, ReportingMode, SlideMeta


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


def _fields(n):
    """n field regions (rect cells inside the 160x120 image), field_index 1..n."""
    out = []
    for i in range(n):
        x0 = 5 + (i % 8) * 18
        y0 = 5 + (i // 8) * 18
        r = counting(f"f{i + 1}", rect(x0, y0, x0 + 15, y0 + 15))
        r.field_index = i + 1
        out.append(r)
    return out


def test_slide_csv_gates_lab_abundance_and_never_hardcodes_volume():
    objs = [obj(f"o{i}", 8 + (i % 2) * 18, 8, ObjectClass.yellow) for i in range(3)]  # 2 in f1, 1 in f2
    st = make_state(objs, _fields(2))
    a = st.analysis
    a.slide.concentration_formula_confirmed = True
    s = rows(exporter.slide_csv(a))[0]
    assert s["concentration_status"].startswith("inputs incomplete: ")
    for m in ("sample_volume_ul", "field_area", "grid_area", "lab_series"):
        assert m in s["concentration_status"]
    assert s["red_only"] == "0" and s["sample_volume_ul"] == ""
    a.slide = SlideMeta(sample_volume_ul=2.5, field_area=2.9, grid_area=0.1, concentration_factor=2.0,
                        lab_series=[LabSeries(name="T7A", count_type=LabCountType.yellow)])
    s = rows(exporter.slide_csv(a))[0]  # all inputs present but formula NOT confirmed
    assert s["concentration_status"] == "inputs incomplete: formula/geometry not confirmed by researcher"
    assert s["abundance_per_ml_T7A"] == "" and s["concentration_formula"] == ""
    a.slide.concentration_formula_confirmed = True
    s = rows(exporter.slide_csv(a))[0]
    assert s["concentration_status"].startswith("computed")
    # mean(2, 1) = 1.5 ; * (2.9/0.1) / 2.5 * 1000 ; the concentration factor (2.0) is NOT applied
    assert float(s["abundance_per_ml_T7A"]) == pytest.approx(1.5 * 29 / 2.5 * 1000)
    assert s["n_fields_T7A"] == "2" and s["lab_series"] == "T7A=yellow" and s["concentration_factor"] == "2.0"
    assert "est_conc_per_ml_red_only" not in s and "sample_total_area_px2" not in s
    assert "CONCENTRATION FACTOR" not in s["concentration_formula"] and "NOT" in s["concentration_formula"]
    # same inputs but unconfirmed mapping => withheld
    st2 = make_state([obj("r", 10, 10)], _fields(1), mapped=False)
    st2.analysis.slide = a.slide.model_copy(deep=True)
    s = rows(exporter.slide_csv(st2.analysis))[0]
    assert "confirmed_channel_mapping" in s["concentration_status"] and s["abundance_per_ml_T7A"] == ""
    # no field region at all => field_regions missing
    st3 = make_state([obj("r", 10, 10)], [counting()])
    st3.analysis.slide = a.slide.model_copy(deep=True)
    assert "field_regions" in rows(exporter.slide_csv(st3.analysis))[0]["concentration_status"]
    # provenance carries the series mapping and the field -> region table
    prov = json.loads(exporter.provenance_json(st))
    assert prov["lab"]["lab_series"][0]["count_type"] == "yellow"
    assert [f["region_id"] for f in prov["lab"]["series"]["T7A"]["fields"]] == ["f1", "f2"]
    assert [f["count"] for f in prov["lab"]["series"]["T7A"]["fields"]] == [2, 1]


def test_final_vs_draft_labels_and_no_confidence_interval():
    st = state()
    a = st.analysis
    assert exporter.export_filename(a, "objects") == "t_objects_DRAFT.csv"
    zdraft = zipfile.ZipFile(io.BytesIO(exporter.bundle_zip(st)))
    assert zdraft.read("README.txt").decode().startswith("DRAFT")
    assert "objects_DRAFT.csv" in zdraft.namelist()
    # no CI column anywhere
    blob = " ".join(zdraft.read(n).decode().lower() for n in zdraft.namelist() if n.endswith(".csv"))
    assert "ci_" not in blob and "confidence" not in blob and "interval" not in blob
    readme = zdraft.read("README.txt").decode()
    assert "No confidence interval is provided" in readme and "Poisson" in readme
    assert "LAB FORMULA" in readme and "mean(field counts)" in readme and "NOT part of this formula" in readme
    st.analysis.finalized_at = "2026-01-01T00:00:00Z"
    assert exporter.export_filename(a, "objects") == "t_objects_FINAL.csv"
    assert rows(exporter.objects_csv(a))[0]["status"] == "FINAL"
    assert exporter.readme_text(a).startswith("FINAL")


def test_provenance_content():
    st = state()
    st.analysis.reporting_mode = ReportingMode.channel_inclusive
    p = json.loads(exporter.provenance_json(st))
    for k in ["schema_version", "app_version", "package_versions_at_detection", "package_versions_at_export", "reporting_mode", "provenance", "image", "regions",
              "detection_parameters", "action_log_summary", "alignment_applied_to_green_px", "status"]:
        assert k in p
    assert p["reporting_mode"] == "channel_inclusive" and p["status"] == "DRAFT"
    assert p["image"]["sha256"] == {"a.tif": "0" * 64} and p["image"]["bit_depth"] == 16
    assert [c["role"] for c in p["image"]["channel_mapping"]] == ["red", "green"]
    assert p["image"]["channel_mapping"][0]["role_confirmed"] is True
    assert p["action_log_summary"]["by_type"] == {"delete": 1}
    assert "numpy" in p["package_versions_at_export"] and "package_versions" not in p


def _conc_state(**slide):
    objs = [obj(f"r{i}", 8 + 3 * i, 12) for i in range(4)]  # 4 red objects, all in field 1
    st = make_state(objs, _fields(2))
    base = dict(sample_volume_ul=2.0, concentration_factor=3.0, field_area=2.0, grid_area=0.5,
                lab_series=[LabSeries(name="S", count_type=LabCountType.red_only)],
                concentration_formula_confirmed=True)
    base.update(slide)
    st.analysis.slide = SlideMeta(**base)
    return st.analysis


def test_lab_abundance_formula_and_blank_vs_zero():
    a = _conc_state()
    s = rows(exporter.slide_csv(a))[0]
    # fields: f1 = 4, f2 = 0 (zero is a value, counted in the mean) -> mean 2 ; * (2/0.5) / 2 * 1000
    assert float(s["abundance_per_ml_S"]) == pytest.approx(2 * 4 / 2 * 1000)
    assert s["n_fields_S"] == "2" and s["concentration_status"].startswith("computed")
    assert s["red_only"] == "4"  # raw counts always present
    # a single field region: blank for the missing one, mean over 1 field = 4
    st = make_state([obj(f"r{i}", 8 + 3 * i, 12) for i in range(4)], _fields(1))
    st.analysis.slide = a.slide.model_copy(deep=True)
    s1 = rows(exporter.slide_csv(st.analysis))[0]
    assert float(s1["abundance_per_ml_S"]) == pytest.approx(4 * 4 / 2 * 1000)
    # the concentration factor does not change the lab formula
    a.slide.concentration_factor = 30.0
    assert float(rows(exporter.slide_csv(a))[0]["abundance_per_ml_S"]) == pytest.approx(4000)


def test_concentration_status_lists_every_missing_input_and_has_no_ci():
    a = _conc_state()
    a.slide.field_area = None
    a.slide.sample_volume_ul = None
    s = rows(exporter.slide_csv(a))[0]
    assert s["concentration_status"] == "inputs incomplete: sample_volume_ul, field_area"
    assert s["abundance_per_ml_S"] == "" and s["red_only"] == "4"
    a.slide.concentration_formula_confirmed = False
    st = rows(exporter.slide_csv(a))[0]["concentration_status"]
    assert st.startswith("inputs incomplete: formula/geometry not confirmed") and "field_area" in st


def test_readme_wording_no_overclaim():
    text = " ".join(exporter.readme_text(state().analysis).split())
    assert "each physical object counted once" not in text
    assert "used at most once in the one-to-one association" in text and "need review" in text
    assert "LAB FORMULA" in text and "field_index" in text and "lab_series" in text
