"""Debris-excluded convenience counts (counts + CSV exports)."""
import csv
import io

from fakes import counting, make_state, obj, rect
from polony_app.export import exporter
from polony_app.review.counts import compute_counts
from polony_app.schemas import ObjectClass, ReviewReason


def test_excl_debris_counts_and_csv():
    d = [ReviewReason.debris_like]
    objs = [obj("a", 20, 20, ObjectClass.red), obj("b", 21, 20, ObjectClass.red, review_reasons=d),
            obj("c", 22, 20, ObjectClass.yellow, review_reasons=d), obj("d", 23, 20, ObjectClass.yellow),
            obj("e", 24, 20, ObjectClass.green, review_reasons=d), obj("f", 25, 20, ObjectClass.green),
            obj("g", 26, 20, ObjectClass.red, review_reasons=d, deleted=True),
            obj("h", 27, 20, ObjectClass.red, review_reasons=[ReviewReason.weak_signal])]
    st = make_state(objs, [counting("c1", rect(5, 5, 100, 100))])
    c = compute_counts(st.analysis)
    r, s = c["regions"][0], c["slide"]
    for k in (r, s):
        assert (k.red_positive, k.green_positive) == (5, 4)
        assert k.red_positive_excl_debris == 3      # a, d, h
        assert k.green_positive_excl_debris == 2    # d, f
    for text in (exporter.regions_csv(st.analysis), exporter.slide_csv(st.analysis)):
        row = list(csv.DictReader(io.StringIO(text)))[0]
        assert row["red_positive_excl_debris"] == "3" and row["green_positive_excl_debris"] == "2"
    assert "objects flagged debris_like remain counted elsewhere" in exporter.readme_text(st.analysis)
