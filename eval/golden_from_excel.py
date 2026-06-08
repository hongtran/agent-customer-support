import json, sys
import openpyxl


def normalize_class(raw: str) -> str:
    s = (raw or "").lower()
    if "hướng dẫn" in s:
        return "how_to"
    if "nâng cấp" in s or "bổ sung" in s or "cập nhật" in s:
        return "feature"
    return "unknown"


def build(xlsx_path: str, out_path: str = "eval/golden.json") -> int:
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb.active
    items = []
    for row in ws.iter_rows(min_row=4, values_only=True):
        content, phanloai = row[5], row[6]
        if content and str(content).strip():
            cls = normalize_class(str(phanloai or ""))
            if cls != "unknown":
                items.append({"request": str(content).strip(), "expected_class": cls})
    with open(out_path, "w") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return len(items)


if __name__ == "__main__":
    n = build(sys.argv[1])
    print(f"wrote {n} golden items to eval/golden.json")
