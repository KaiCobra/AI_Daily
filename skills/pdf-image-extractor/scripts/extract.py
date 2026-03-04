#!/usr/bin/env python3
"""
extract.py — PDF Image Extractor
Part of the pdf-image-extractor Manus Skill.

Extracts embedded images from a PDF file and saves them to a specified directory.
Also prints the text content of each page.

Usage:
    python3 extract.py <input_pdf> <output_dir> <img_format> <img_quality>

Arguments:
    input_pdf   Path to the source PDF file.
    output_dir  Directory where extracted images will be saved.
    img_format  Output image format: png, jpeg, or webp.
    img_quality Image quality (1-100). Use 100 for lossless PNG.

Example:
    python3 extract.py ./document.pdf ./output_images/ png 100
"""

import pdfplumber
import fitz
import os
import io
import sys
import argparse
from PIL import Image


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Extract embedded images from a PDF file and save them to a directory. "
            "Also prints the text content of each page."
        )
    )
    parser.add_argument("input_file", help="Path to the source PDF file.")
    parser.add_argument("output_dir", help="Directory where extracted images will be saved.")
    parser.add_argument(
        "img_format",
        choices=["png", "jpeg", "webp"],
        help="Output image format: png, jpeg, or webp.",
    )
    parser.add_argument(
        "img_quality",
        type=int,
        help="Image quality from 1 to 100 (100 = lossless for PNG).",
    )
    return parser.parse_args()


def remove_letters(text: str) -> str:
    """Return only the numeric characters from a string."""
    return "".join(c for c in text if not c.isalpha())


def sanitize_filename(text: str) -> str:
    """Sanitize a string so it can be safely used as a filename."""
    text = text.replace("\n", "").strip()
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        text = text.replace(char, "")
    text = " ".join(text.split())
    return text


def resize_image(image: Image.Image) -> Image.Image:
    """
    Resize an image so that neither dimension exceeds 70% of the larger side.

    This reduces file size while preserving aspect ratio using bicubic resampling.
    """
    max_size = int(max(image.width, image.height) * 0.7)

    if image.width > max_size or image.height > max_size:
        if image.width > image.height:
            aspect_ratio = image.height / image.width
            new_width = max_size
            new_height = int(max_size * aspect_ratio)
        else:
            aspect_ratio = image.width / image.height
            new_height = max_size
            new_width = int(max_size * aspect_ratio)

        image = image.resize((new_width, new_height), Image.BICUBIC)

    return image


def save_images_from_page(
    document: fitz.Document,
    page_number: int,
    product_reference: str,
    output_dir: str,
    img_format: str,
    img_quality: int,
) -> list[str]:
    """
    Extract and save all qualifying images from a single PDF page.

    Images smaller than 500×500 pixels are skipped (likely icons or decorative elements).
    Each saved image is resized to 70% of its larger dimension.

    Returns:
        A list of file paths for the saved images.
    """
    saved_images = []
    page = document.load_page(page_number)
    images = page.get_images(full=True)

    for img_index, img in enumerate(images):
        xref = img[0]
        base_image = document.extract_image(xref)
        image_bytes = base_image["image"]

        image = Image.open(io.BytesIO(image_bytes))

        # Skip images that are too small (icons, logos, decorative elements)
        if image.width < 500 or image.height < 500:
            continue

        # Resize to reduce file size
        image = resize_image(image)

        # Build a safe filename from the page's text content
        safe_reference = sanitize_filename(product_reference)
        image_filename = os.path.join(
            output_dir, f"{safe_reference}_{page_number + 1}.{img_format}"
        )

        image.save(image_filename, img_format.upper(), quality=img_quality)
        saved_images.append(image_filename)

    return saved_images


def main():
    args = parse_arguments()

    pdf_path = args.input_file
    output_dir = args.output_dir
    img_format = args.img_format
    img_quality = args.img_quality

    # Validate inputs
    if not os.path.isfile(pdf_path):
        print(f"ERROR: PDF file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(output_dir):
        print(f"Output directory does not exist. Creating: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    if not (1 <= img_quality <= 100):
        print("ERROR: img_quality must be between 1 and 100.", file=sys.stderr)
        sys.exit(1)

    print(f"PDF:          {pdf_path}")
    print(f"Output dir:   {output_dir}")
    print(f"Image format: {img_format.upper()}")
    print(f"Quality:      {img_quality}")
    print("=" * 60)

    total_images_saved = 0

    # Open the document with fitz for image extraction
    document = fitz.open(pdf_path)

    # Use pdfplumber for text extraction alongside fitz for images
    with pdfplumber.open(pdf_path) as pdf:
        for index, page in enumerate(pdf.pages):
            print(f"\nPage {index + 1}:")

            # Extract and display text
            text = page.extract_text() or ""
            print(text.strip())
            print("-" * 50)

            # Extract and save images
            saved = save_images_from_page(
                document=document,
                page_number=index,
                product_reference=remove_letters(text),
                output_dir=output_dir,
                img_format=img_format,
                img_quality=img_quality,
            )

            for img_path in saved:
                print(f"  Image saved: {img_path}")

            total_images_saved += len(saved)
            print("=" * 60)

    document.close()

    print(f"\nDone. Total images extracted: {total_images_saved}")
    print(f"Images saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
