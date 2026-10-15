"""Embedded researcher marks of the real AI-A slide vs the Counts sheet (skipped without the local TIFFs)."""
import pytest

from legacy_fixtures import REPO
from polony_app.fields.reference import load_reference
from polony_app.imaging.ingest import ingest_files

DIR = REPO / "resources" / "research_tiffs"
AI_A = DIR / "29-09-2026_T7_AB_AI-A.tif"
SHEET_AI_A = [13, 29, 15, 15, 16, 25, 23, 15, 12, 15, 22, 16, 15, 22, 13, 19, 23, 24]   # 'Clade B', slide AI/A, fields 1..18


@pytest.mark.skipif(not AI_A.exists(), reason="real research TIFFs not present")
def test_ai_a_marks_equal_the_sheet_by_ordinal_numbering(tmp_path):
    meta = ingest_files([(AI_A.name, AI_A)], tmp_path)
    assert [c.suggested_role.value for c in meta.channels] == ["red", "green"]
    ref = load_reference(tmp_path, meta)
    assert ref.available and ref.grid.cols == 4 and ref.grid.rows == 7
    ordinal = ref.counts("ordinal", plane=0)
    assert [ordinal[k] for k in range(1, 19)] == SHEET_AI_A
    index = ref.counts("index", plane=0)
    # AI-A has no counter 3: 'index' numbering (counter + 1) is shifted from field 4 on and warns about the gap
    assert [index.get(k, 0) for k in range(1, 4)] == SHEET_AI_A[:3] and index.get(4, 0) == 0
    assert any("not contiguous" in w for w in ref.warnings)
