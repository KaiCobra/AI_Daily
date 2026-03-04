---
name: pdf-image-extractor
description: >
  Extract embedded images from PDF files and save them to a local directory.
  Supports multiple output formats (PNG, JPEG, WEBP) and configurable quality settings.
  Also extracts and displays text content from each PDF page.
  Activate when the user asks to extract images from a PDF, save PDF images, or process PDF files for image content.
compatibility: Requires Python 3.x with pdfplumber, PyMuPDF (fitz), and Pillow installed.
metadata:
  author: KaiCobra
  version: "1.0"
  source: https://github.com/gsmatheus/pdf-image-extractor
---

# PDF Image Extractor

This skill extracts embedded images from PDF files using PyMuPDF and pdfplumber. It also prints the text content of each page as a side effect.

## First-time setup

Install the required Python dependencies before running the extractor:

```bash
bash scripts/setup.sh
```

Verify the installation succeeded:

```bash
python3 -c "import fitz, pdfplumber, PIL; print('All dependencies installed.')"
```

## Basic usage

Run the extraction script with the following arguments:

```bash
python3 scripts/extract.py <input_pdf> <output_dir> <img_format> <img_quality>
```

| Argument | Description | Example |
|---|---|---|
| `input_pdf` | Path to the source PDF file | `./document.pdf` |
| `output_dir` | Directory where extracted images will be saved | `./output_images/` |
| `img_format` | Output image format: `png`, `jpeg`, or `webp` | `png` |
| `img_quality` | Image quality from 1–100 (100 = lossless for PNG) | `95` |

### Example: Extract as PNG at full quality

```bash
python3 scripts/extract.py ./document.pdf ./output_images/ png 100
```

### Example: Extract as JPEG at 85% quality

```bash
python3 scripts/extract.py ./document.pdf ./output_images/ jpeg 85
```

### Example: Extract as WEBP (efficient compression)

```bash
python3 scripts/extract.py ./document.pdf ./output_images/ webp 90
```

## What the script does

1. Opens the PDF using both PyMuPDF (for image extraction) and pdfplumber (for text extraction).
2. Iterates through every page of the PDF.
3. For each page, extracts all embedded images using PyMuPDF's `get_images()` method.
4. Skips images smaller than 500×500 pixels (likely icons or decorative elements).
5. Resizes each image to 70% of its larger dimension using bicubic resampling.
6. Saves each image to the output directory with a filename derived from the page's text content.
7. Prints the extracted text from each page to standard output.

## Output file naming

Images are named using the numeric characters extracted from the page text, combined with the page number:

```
{numeric_text_from_page}_{page_number}.{format}
```

For example, a product catalog page containing "Item #12345" on page 3 would produce:
```
12345_3.png
```

## Important notes

- Only images with both width and height **≥ 500 pixels** are extracted. Smaller images are silently skipped.
- The output directory must exist before running the script. Create it with `mkdir -p <output_dir>` if needed.
- For PDFs with no embedded images (e.g., scanned documents rendered as image layers), this tool will not extract anything. Use a PDF-to-image renderer instead.
- If the PDF is password-protected, PyMuPDF will raise an error. Decrypt the PDF first.

## Troubleshooting

If the script exits with a `ModuleNotFoundError`, re-run `scripts/setup.sh` to reinstall dependencies.

If no images are saved despite the PDF containing visible images, the images may be smaller than the 500-pixel threshold. You can lower this threshold by editing line 92 of `scripts/extract.py`:

```python
if image.width < 500 or image.height < 500:
```

Change `500` to a smaller value (e.g., `100`) to capture smaller images.
