"""The detect/preview calls must pass real channel indices and ONLY the user-confirmed roles to the engines."""
from polony_app.api.main import _detector_kwargs
from polony_app.schemas import (Analysis, AnalysisStatus, ChannelInfo, ChannelRole, ImageKind, ImageMeta)


def _analysis(roles, confirmed):
    chans = [ChannelInfo(index=i, source_file="x.tif", page=i, role=r, role_confirmed=confirmed)
             for i, r in enumerate(roles)]
    img = ImageMeta(kind=ImageKind.raw_two_channel, height=10, width=10, dtype="uint16", bit_depth=16,
                    sha256={"x.tif": "ab"}, channels=chans)
    return Analysis(id="a1", name="n", created_at="t", updated_at="t", status=AnalysisStatus.ready, image=img)


def _fn(analysis_id="", pages=(0, 1), roles=("red", "green"), image_sha256=None):  # noqa: ARG001
    return None


def test_confirmed_mapping_passes_pages_and_roles():
    kw = _detector_kwargs(_fn, "a1", _analysis([ChannelRole.green, ChannelRole.red], True))
    assert kw["pages"] == (1, 0)                    # red is channel 1, green is channel 0 (swapped mapping)
    assert kw["roles"] == ("red", "green")
    assert kw["analysis_id"] == "a1" and kw["image_sha256"] == {"x.tif": "ab"}


def test_unconfirmed_mapping_never_passes_roles():
    kw = _detector_kwargs(_fn, "a1", _analysis([ChannelRole.red, ChannelRole.green], False))
    assert "roles" not in kw and "pages" not in kw   # colour is never inferred from page order


def test_older_run_detection_signature_still_works():
    def old(red, green):  # noqa: ARG001
        return None
    assert _detector_kwargs(old, "a1", _analysis([ChannelRole.red, ChannelRole.green], True)) == {}
