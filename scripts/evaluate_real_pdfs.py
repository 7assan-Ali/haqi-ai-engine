import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.pdf_loader import load_pdf
from app.rag.text_normalizer import normalize_arabic_text

folder = Path("Pdfs/DataSet/التشريعات و الاحوال المدنية")

pdf_file = next(
    file for file in folder.glob("*.pdf")
    if "١٦٥" in file.name
)

document = load_pdf(str(pdf_file))

for page_number in [1, 2]:
    page = document.pages[page_number]

    print(f"\n{'=' * 70}")
    print(f"PAGE {page.page_number} - NORMALIZED TEXT")
    print(normalize_arabic_text(page.text))