#!/usr/bin/env python3
"""
PDF Image Extractor Script

Extracts images from PDF files and classifies them by type (figures, tables, formulas).
Intelligently names images based on paper conventions (fig, figure, tab, table, etc.).
"""

import os
import sys
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import fitz  # PyMuPDF
from PIL import Image
import io


class PDFImageExtractor:
    """Extract and classify images from PDF documents."""
    
    # Patterns to identify image types from surrounding text
    FIGURE_PATTERNS = [
        r'\b(?:fig|figure|fig\.)\s*\.?\s*(\d+[a-z]?)',
        r'\b(?:Fig|Figure|FIG)\s*\.?\s*(\d+[a-z]?)',
    ]
    
    TABLE_PATTERNS = [
        r'\b(?:tab|table|tbl|tab\.)\s*\.?\s*(\d+[a-z]?)',
        r'\b(?:Tab|Table|TBL|Tbl)\s*\.?\s*(\d+[a-z]?)',
    ]
    
    FORMULA_PATTERNS = [
        r'\b(?:eq|equation|formula|eqn|eqn\.)\s*\.?\s*(\d+[a-z]?)',
        r'\b(?:Eq|Equation|Formula|Eqn|EQN)\s*\.?\s*(\d+[a-z]?)',
    ]
    
    def __init__(self, pdf_path: str, output_dir: str):
        """
        Initialize the extractor.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images
        """
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.doc = None
        self.images_data = []
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    def open_pdf(self) -> None:
        """Open the PDF document."""
        try:
            self.doc = fitz.open(self.pdf_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF: {e}")
    
    def close_pdf(self) -> None:
        """Close the PDF document."""
        if self.doc:
            self.doc.close()
    
    def extract_text_around_image(self, page_num: int, rect: fitz.Rect) -> str:
        """
        Extract text near an image to help classify it.
        
        Args:
            page_num: Page number (0-indexed)
            rect: Rectangle of the image
            
        Returns:
            Text near the image
        """
        if not self.doc:
            return ""
        
        page = self.doc[page_num]
        
        # Expand search area around image
        expanded_rect = rect + 100
        
        # Get text from expanded area
        text = page.get_text("text", clip=expanded_rect)
        return text.lower()
    
    def classify_image(self, page_num: int, rect: fitz.Rect, image_index: int) -> Tuple[str, Optional[str]]:
        """
        Classify image type based on surrounding text.
        
        Args:
            page_num: Page number (0-indexed)
            rect: Rectangle of the image
            image_index: Index of image on page
            
        Returns:
            Tuple of (category, label) where category is 'figures', 'tables', 'formulas'
            and label is the identified reference (e.g., 'fig_1', 'table_2')
        """
        text = self.extract_text_around_image(page_num, rect)
        
        # Check for table patterns
        for pattern in self.TABLE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                label = f"table_{match.group(1)}"
                return ("tables", label)
        
        # Check for formula patterns
        for pattern in self.FORMULA_PATTERNS:
            match = re.search(pattern, text)
            if match:
                label = f"formula_{match.group(1)}"
                return ("formulas", label)
        
        # Default to figures
        for pattern in self.FIGURE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                label = f"fig_{match.group(1)}"
                return ("figures", label)
        
        # If no pattern found, use generic naming
        return ("figures", None)
    
    def extract_images(self) -> Dict[str, List[str]]:
        """
        Extract all images from PDF and classify them.
        
        Returns:
            Dictionary mapping category to list of saved file paths
        """
        if not self.doc:
            self.open_pdf()
        
        # Create output directories
        categories = ["figures", "tables", "formulas"]
        for category in categories:
            (self.output_dir / category).mkdir(parents=True, exist_ok=True)
        
        results = {category: [] for category in categories}
        
        # Track counters for unnamed images
        counters = {category: 0 for category in categories}
        
        # Iterate through all pages
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            
            # Get all images on the page
            image_list = page.get_images()
            
            for img_index, img_ref in enumerate(image_list):
                try:
                    # Extract image
                    xref = img_ref[0]
                    pix = fitz.Pixmap(self.doc, xref)
                    
                    # Get image bbox
                    try:
                        rect = page.get_image_bbox(img_ref)
                    except:
                        # Fallback if bbox is not available
                        rect = fitz.Rect(0, 0, 100, 100)
                    
                    # Classify the image
                    category, label = self.classify_image(page_num, rect, img_index)
                    
                    # Generate filename
                    if label:
                        filename = f"{label}.png"
                    else:
                        counters[category] += 1
                        filename = f"{category[:-1]}_{page_num + 1}_{counters[category]}.png"
                    
                    # Save image
                    output_path = self.output_dir / category / filename
                    
                    # Convert to RGB if necessary
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        pix.save(str(output_path))
                    else:  # RGBA
                        pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                        pix_rgb.save(str(output_path))
                    
                    results[category].append(str(output_path))
                    print(f"✓ Extracted: {filename} (page {page_num + 1})")
                    
                except Exception as e:
                    print(f"✗ Failed to extract image on page {page_num + 1}: {e}", file=sys.stderr)
        
        return results
    
    def print_summary(self, results: Dict[str, List[str]]) -> None:
        """Print extraction summary."""
        print("\n" + "="*60)
        print("PDF Image Extraction Summary")
        print("="*60)
        
        total = sum(len(images) for images in results.values())
        print(f"Total images extracted: {total}\n")
        
        for category, images in results.items():
            if images:
                print(f"{category.upper()}: {len(images)} images")
                for img_path in images:
                    print(f"  - {Path(img_path).name}")
        
        print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract images from PDF and classify by type"
    )
    parser.add_argument(
        "pdf_file",
        help="Path to the PDF file"
    )
    parser.add_argument(
        "-o", "--output",
        default="extracted_images",
        help="Output directory (default: extracted_images)"
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Don't print summary after extraction"
    )
    
    args = parser.parse_args()
    
    try:
        extractor = PDFImageExtractor(args.pdf_file, args.output)
        extractor.open_pdf()
        
        print(f"Processing: {args.pdf_file}")
        print(f"Output directory: {args.output}\n")
        
        results = extractor.extract_images()
        
        if not args.no_summary:
            extractor.print_summary(results)
        
        extractor.close_pdf()
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
