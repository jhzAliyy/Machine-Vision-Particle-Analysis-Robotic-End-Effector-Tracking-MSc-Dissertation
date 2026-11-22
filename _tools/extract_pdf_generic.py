import sys, os
import fitz  # PyMuPDF

def extract(pdf_path, out_path=None):
    if out_path is None:
        out_path = os.path.splitext(pdf_path)[0] + "_text.txt"
    doc = fitz.open(pdf_path)
    n = doc.page_count
    parts = []
    for i, page in enumerate(doc):
        parts.append(f"==================== PAGE {i+1} ====================")
        parts.append(page.get_text("text"))
    doc.close()
    text = "\n".join(parts)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"OK: {n} pages -> {out_path} ({len(text)} chars)")

if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
