import sys, os
import fitz  # PyMuPDF

def extract(pdf_path, out_path):
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
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # 手册 PDF 在 uk_supervisor/ 层（本脚本位于其下 dissertation/_tools/）
    pdf = os.path.join(base, "H1043Z Individual Project Handbook v2025.pdf")
    out = os.path.join(base, "H1043Z_handbook_text.txt")
    extract(pdf, out)
