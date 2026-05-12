import os
import re
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image

# ============================================================
# CONFIG
# ============================================================

PDF_PATH = "input.pdf"
OUTPUT_DIR = "cropped_tables"

# Number of text lines to include above/below table
LINES_ABOVE = 3
LINES_BELOW = 3

# Keywords to identify labels
LABEL_PATTERNS = [
    r"^table\s+\d+",
    r"^table\s+[A-Za-z0-9.\-]+",
    r"^figure\s+\d+",
    r"^fig\.\s*\d+",
]

# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_filename(text):
    """
    Convert caption text into filesystem-safe filename.
    """
    text = re.sub(r"[^\w\s\-]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:120]


def is_label(text):
    """
    Check whether a line looks like a table/figure label.
    """
    text = text.strip().lower()

    for pattern in LABEL_PATTERNS:
        if re.match(pattern, text):
            return True

    return False


def rect_union(rects):
    """
    Merge multiple rectangles into one.
    """
    x0 = min(r.x0 for r in rects)
    y0 = min(r.y0 for r in rects)
    x1 = max(r.x1 for r in rects)
    y1 = max(r.y1 for r in rects)

    return fitz.Rect(x0, y0, x1, y1)


# ============================================================
# OPEN PDF
# ============================================================

doc = fitz.open(PDF_PATH)

# ============================================================
# PROCESS EACH PAGE
# ============================================================

for page_num in range(len(doc)):

    page = doc[page_num]

    print(f"\nProcessing page {page_num + 1}")

    # --------------------------------------------------------
    # FIND TABLES
    # --------------------------------------------------------

    tables = page.find_tables()

    if not tables.tables:
        print("No tables found")
        continue

    # --------------------------------------------------------
    # EXTRACT TEXT LINES
    # --------------------------------------------------------

    text_dict = page.get_text("dict")

    lines = []

    for block in text_dict["blocks"]:

        if block["type"] != 0:
            continue

        for line in block["lines"]:

            spans = line["spans"]

            if not spans:
                continue

            line_text = " ".join(span["text"] for span in spans).strip()

            if not line_text:
                continue

            bbox = fitz.Rect(spans[0]["bbox"])

            for span in spans[1:]:
                bbox |= fitz.Rect(span["bbox"])

            lines.append({
                "text": line_text,
                "bbox": bbox
            })

    # Sort lines vertically
    lines = sorted(lines, key=lambda x: x["bbox"].y0)

    # --------------------------------------------------------
    # PROCESS EACH TABLE
    # --------------------------------------------------------

    for table_idx, table in enumerate(tables.tables):

        table_rect = fitz.Rect(table.bbox)

        # ====================================================
        # FIND LABEL ABOVE TABLE
        # ====================================================

        label_index = None

        for idx, line in enumerate(lines):

            # line above table
            if line["bbox"].y1 <= table_rect.y0:

                if is_label(line["text"]):
                    label_index = idx

        # ====================================================
        # FIND LABEL BELOW TABLE IF NOT FOUND ABOVE
        # ====================================================

        if label_index is None:

            for idx, line in enumerate(lines):

                if line["bbox"].y0 >= table_rect.y1:

                    if is_label(line["text"]):
                        label_index = idx
                        break

        # ====================================================
        # COLLECT LABEL + CONTEXT LINES
        # ====================================================

        selected_rects = [table_rect]

        label_text = f"table_{table_idx+1}"

        if label_index is not None:

            start_idx = max(0, label_index - LINES_ABOVE)
            end_idx = min(len(lines), label_index + LINES_BELOW + 1)

            context_lines = lines[start_idx:end_idx]

            for item in context_lines:
                selected_rects.append(item["bbox"])

            label_text = lines[label_index]["text"]

        # ====================================================
        # MERGE RECTANGLES
        # ====================================================

        final_rect = rect_union(selected_rects)

        # Add padding
        padding = 10

        final_rect.x0 -= padding
        final_rect.y0 -= padding
        final_rect.x1 += padding
        final_rect.y1 += padding

        # ====================================================
        # RENDER CROPPED IMAGE
        # ====================================================

        pix = page.get_pixmap(
            clip=final_rect,
            matrix=fitz.Matrix(3, 3)  # High resolution
        )

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        # ====================================================
        # SAVE IMAGE
        # ====================================================

        filename = (
            f"{clean_filename(label_text)}"
            f"_page_{page_num + 1}.png"
        )

        output_path = os.path.join(OUTPUT_DIR, filename)

        image.save(output_path)

        print(f"Saved: {output_path}")

print("\nDone.")