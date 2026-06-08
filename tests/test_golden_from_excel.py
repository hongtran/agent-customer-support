from eval.golden_from_excel import normalize_class


def test_normalize_class():
    assert normalize_class("Hướng dẫn sử dụng") == "how_to"
    assert normalize_class("Nâng cấp") == "feature"
    assert normalize_class("Cập nhật") == "feature"
    assert normalize_class("") == "unknown"
