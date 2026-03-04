#!/usr/bin/env bash
# setup.sh — Install Python dependencies for the PDF Image Extractor skill.
# Run this script once before using the skill for the first time.

set -euo pipefail

echo "=== PDF Image Extractor: Dependency Setup ==="

# Check if Python 3 is available
if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python 3 is not installed. Please install Python 3.x first." >&2
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1)
echo "Found: $PYTHON_VERSION"

# Install required packages
echo ""
echo "Installing required Python packages..."
echo "  - pdfplumber  (PDF text and layout extraction)"
echo "  - pymupdf     (PDF image extraction via fitz)"
echo "  - Pillow      (image processing)"
echo ""

if command -v pip3 &>/dev/null; then
    pip3 install --quiet pdfplumber pymupdf pillow
elif command -v pip &>/dev/null; then
    pip install --quiet pdfplumber pymupdf pillow
else
    echo "ERROR: pip is not available. Please install pip first." >&2
    exit 1
fi

echo ""
echo "Verifying installation..."
python3 -c "
import fitz
import pdfplumber
from PIL import Image
print('  fitz (PyMuPDF):', fitz.__version__)
print('  pdfplumber:    OK')
print('  Pillow:        OK')
print('')
print('All dependencies installed successfully.')
"
