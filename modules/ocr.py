import re
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter


def extract_receipt_details(image):

    # ============================================================
    # OPEN IMAGE
    # ============================================================

    img = Image.open(image)

    # ============================================================
    # IMAGE PREPROCESSING
    # ============================================================

    img = img.convert("L")

    img = ImageEnhance.Contrast(img).enhance(2)

    img = img.filter(
        ImageFilter.SHARPEN
    )

    # ============================================================
    # OCR
    # ============================================================

    text = pytesseract.image_to_string(
        img,
        config="--psm 6"
    )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # ============================================================
    # DEFAULT VALUES
    # ============================================================

    description = "Receipt"

    amount = 0.0

    # ============================================================
    # MERCHANT / DESCRIPTION
    # ============================================================

    # Try to find the first meaningful text line.
    # Ignore lines containing prices or quantities.

    for line in lines:

        cleaned = line.strip()

        # Skip very short lines
        if len(cleaned) < 4:
            continue

        # Skip lines containing TOTAL / CASH / CHANGE
        upper_line = cleaned.upper()

        if any(
            word in upper_line
            for word in [
                "TOTAL",
                "CASH",
                "CHANGE",
                "SUBTOTAL",
                "TAX"
            ]
        ):
            continue

        # Skip lines that are mostly numbers
        digit_count = sum(
            char.isdigit()
            for char in cleaned
        )

        if digit_count > len(cleaned) / 2:
            continue

        # First meaningful line = merchant
        description = cleaned
        break

    # ============================================================
    # TOTAL AMOUNT
    # ============================================================

    # First priority: TOTAL line

    for line in lines:

        if "TOTAL" in line.upper():

            nums = re.findall(
                r"\d+(?:\.\d{1,2})?",
                line
            )

            if nums:

                try:

                    amount = float(
                        nums[-1]
                    )

                    break

                except ValueError:
                    pass

    # ============================================================
    # FALLBACK AMOUNT EXTRACTION
    # ============================================================

    if amount == 0:

        numbers = []

        for line in lines:

            nums = re.findall(
                r"\d+\.\d{1,2}",
                line
            )

            for num in nums:

                try:
                    numbers.append(
                        float(num)
                    )

                except ValueError:
                    pass

        if numbers:

            # Usually the receipt total is one
            # of the largest decimal values.
            amount = max(numbers)

    # ============================================================
    # RETURN
    # ============================================================

    return description, amount, text