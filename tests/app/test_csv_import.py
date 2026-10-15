from pathlib import Path

import pytest

from polony_app.store.csv_import import parse_csv

# synthetic lab-format CSVs (tests/test_data/lab_templates/make_templates.py)
TEMPLATES = Path(__file__).resolve().parents[1] / "test_data" / "lab_templates"
LAB_CSVS = [TEMPLATES / "T7B_template.csv", TEMPLATES / "T7A_template.csv"]

FIXTURE = (
    "﻿slide ID,SAMPLING DATE,POLONY DATE ,CONCENTRATION FACTOR,SAMPLE VOLUM[ul],field [Pixel^2],grid [Pixel^2],"
    "FAGE ABUNDENCE [fage mL-1],field 1,field 2,field 3,field 4\r\n"
    "S1,15/09/2026,29/09/2026,1,2.5,2.9,0.1,100,1,0,,\r\n"
    "S2,15/09/2026,29/09/2026,1,2.5,2.9,0.1,,0,0,0,0\r\n"
    "\r\n"
    "S3,x,y,z,,,,,,3\r\n"
).encode("utf-8")


def test_blank_is_not_zero_and_raw_kept():
    c = parse_csv(FIXTURE, "t.csv", "id1")
    assert c.header[0] == "slide ID"  # BOM stripped
    assert c.header[2] == "POLONY DATE "  # trailing space preserved verbatim
    assert [r.slide_id for r in c.rows] == ["S1", "S2", "S3"]
    s1, s2, s3 = c.rows
    assert s1.fields == [1, 0, None, None] and s1.raw_fields == ["1", "0", "", ""]
    assert s2.fields == [0, 0, 0, 0]
    assert s3.fields == [None, 3, None, None]  # short row padded with blanks (not zeros)
    assert s1.fields[1] == 0 and s1.fields[1] is not None and s1.fields[2] is None
    assert s1.meta["SAMPLE VOLUM[ul]"] == "2.5" and s1.meta["field [Pixel^2]"] == "2.9"
    assert s2.meta["FAGE ABUNDENCE [fage mL-1]"] == ""  # blank metadata stays blank, not 0
    assert any("blank field cell" in n for n in c.notes)
    assert any("shorter than header" in n for n in c.notes)
    assert c.color_mapping is None


def test_no_color_inferred_from_names():
    c = parse_csv(FIXTURE, "T7A_15-09-2026.csv", "id1")
    assert c.color_mapping is None
    assert any("no colour mapping is inferred" in n for n in c.notes)


def test_sha_and_garbage():
    c = parse_csv(b"", "e.csv", "i")
    assert c.rows == [] and "empty" in c.notes[-1]
    c = parse_csv(b"slide ID,field 1\nA,x\nB,2.0\n", "e.csv", "i")
    assert c.rows[0].fields == [None] and c.rows[0].raw_fields == ["x"]
    assert c.rows[1].fields == [None] and c.rows[1].raw_fields == ["2.0"]  # not silently coerced
    assert sum("not an integer" in n for n in c.notes) == 2


@pytest.mark.skipif(not all(p.exists() for p in LAB_CSVS), reason="lab-format template CSVs absent")
@pytest.mark.parametrize("path", LAB_CSVS, ids=lambda p: p.name)
def test_lab_format_files(path):
    data = path.read_bytes()
    c = parse_csv(data, path.name, "r1")
    assert c.header[0] == "slide ID" and len(c.header) == 9 + 24
    assert len(c.rows) >= 20
    assert c.header[4] == "SAMPLE VOLUM[ul]"
    for r in c.rows:
        assert len(r.fields) == len(r.raw_fields) == 24
        for v, raw in zip(r.fields, r.raw_fields):
            assert (v is None) == (raw.strip() == "")
            if raw.strip() != "":
                assert v == int(raw)
        assert r.meta["SAMPLE VOLUM[ul]"] == "2.5"
        assert r.meta["field [Pixel^2]"] == "2.9" and r.meta["grid [Pixel^2]"] == "0.1"
    # the first T7A row has 18 measured cells (incl. zeros) followed by 6 blanks, which must stay blank
    if "T7A" in path.name:
        first = c.rows[0]
        assert first.slide_id == "S01_A" and first.fields[:3] == [1, 3, 0]
        assert first.fields[18:] == [None] * 6 and first.fields[17] == 0 and first.fields[3] == 0
    assert c.color_mapping is None
