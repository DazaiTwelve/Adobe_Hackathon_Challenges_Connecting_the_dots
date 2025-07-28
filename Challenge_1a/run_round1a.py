#!/usr/bin/env python3
"""
Round 1A: Document Outline Extraction
Extracts title and hierarchical headings (H1, H2, H3) from PDF documents.
"""

import os
import json
import sys
import time
import argparse
from typing import List, Dict, Optional
import fitz  # PyMuPDF
import re

class OutlineExtractor:
    def __init__(self):
        self.heading_patterns = [
            # Numbered headings: 1.1, 1.2, etc.
            r'^\d+(\.\d+)*\s+[A-Z]',
            # All caps headings
            r'^[A-Z\s]{3,}$',
            # Title case with limited words
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$'
        ]
    
    def extract_outline(self, pdf_path: str) -> Dict:
        """
        Extract title and outline from PDF.
        Returns: {"title": str, "outline": [{"level": str, "text": str, "page": int}]}
        """
        try:
            doc = fitz.open(pdf_path)
            if len(doc) > 50:
                print(f"Warning: PDF has {len(doc)} pages, processing first 50")
                doc = doc[:50]
            
            # Extract all text blocks with metadata
            blocks = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_blocks = page.get_text("dict")['blocks']
                
                for block in page_blocks:
                    if block['type'] == 0:  # text block
                        for line in block['lines']:
                            for span in line['spans']:
                                text = span['text'].strip()
                                if text:
                                    blocks.append({
                                        'text': text,
                                        'size': round(span['size']),
                                        'page': page_num + 1,
                                        'bbox': span['bbox']
                                    })
            
            if not blocks:
                return {"title": "", "outline": []}
            
            # Analyze font sizes to determine heading levels
            font_sizes = {}
            for block in blocks:
                size = block['size']
                font_sizes[size] = font_sizes.get(size, 0) + 1
            
            # Sort font sizes descending
            sorted_sizes = sorted(font_sizes.keys(), reverse=True)
            
            # Assign heading levels based on font size hierarchy
            size_to_level = {}
            if len(sorted_sizes) > 0:
                size_to_level[sorted_sizes[0]] = "TITLE"
            if len(sorted_sizes) > 1:
                size_to_level[sorted_sizes[1]] = "H1"
            if len(sorted_sizes) > 2:
                size_to_level[sorted_sizes[2]] = "H2"
            if len(sorted_sizes) > 3:
                size_to_level[sorted_sizes[3]] = "H3"
            
            # Extract title and headings
            title = ""
            outline = []
            
            for block in blocks:
                text = block['text']
                size = block['size']
                page = block['page']
                
                # Skip empty or very short text
                if len(text) < 2:
                    continue
                
                level = size_to_level.get(size)
                
                if level == "TITLE" and not title:
                    # Clean up title
                    title = self._clean_text(text)
                elif level in ["H1", "H2", "H3"]:
                    # Additional validation for headings
                    if self._is_valid_heading(text):
                        outline.append({
                            "level": level,
                            "text": self._clean_text(text),
                            "page": page
                        })
            
            # Fallback: if no title found, use first heading
            if not title and outline:
                title = outline[0]["text"]
                outline = outline[1:]  # Remove from outline since it's now the title
            
            # Fallback: if no headings found, try pattern-based detection
            if not outline:
                outline = self._extract_headings_by_pattern(blocks)
            
            # Ensure we have a title
            if not title:
                title = "Document Title"
            
            return {
                "title": title,
                "outline": outline
            }
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return {"title": "Document Title", "outline": []}
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters that might break JSON
        text = text.replace('\n', ' ').replace('\r', ' ')
        return text
    
    def _is_valid_heading(self, text: str) -> bool:
        """Check if text looks like a valid heading."""
        # Must be at least 2 characters
        if len(text) < 2:
            return False
        
        # Check for common heading patterns
        for pattern in self.heading_patterns:
            if re.match(pattern, text):
                return True
        
        # Check if it's short and starts with capital letter
        if len(text.split()) <= 8 and text[0].isupper():
            return True
        
        return False
    
    def _extract_headings_by_pattern(self, blocks: List[Dict]) -> List[Dict]:
        """Extract headings using pattern matching when font size fails."""
        outline = []
        
        for block in blocks:
            text = block['text']
            page = block['page']
            
            if self._is_valid_heading(text):
                # Determine level based on text characteristics
                level = self._determine_heading_level(text)
                if level:
                    outline.append({
                        "level": level,
                        "text": self._clean_text(text),
                        "page": page
                    })
        
        return outline
    
    def _determine_heading_level(self, text: str) -> Optional[str]:
        """Determine heading level based on text characteristics."""
        # Numbered headings: 1.1, 1.2, etc.
        if re.match(r'^\d+(\.\d+)*\s+[A-Z]', text):
            if text.count('.') == 0:
                return "H1"
            elif text.count('.') == 1:
                return "H2"
            else:
                return "H3"
        
        # All caps headings
        if text.isupper() and len(text) > 2:
            return "H1"
        
        # Title case with limited words
        if text.istitle() and len(text.split()) <= 5:
            return "H2"
        
        return None

def process_pdf(pdf_path: str, output_path: str) -> bool:
    """Process a single PDF and save the outline to JSON."""
    try:
        extractor = OutlineExtractor()
        result = extractor.extract_outline(pdf_path)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save result
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Processed: {os.path.basename(pdf_path)} -> {os.path.basename(output_path)}")
        print(f"   Title: {result['title'][:50]}...")
        print(f"   Headings: {len(result['outline'])} found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")
        return False

def main():
    """Main function to process all PDFs in input directory."""
    parser = argparse.ArgumentParser(description="Round 1A: PDF Outline Extraction")
    parser.add_argument("--input_dir", default="/app/input", help="Input directory containing PDFs")
    parser.add_argument("--output_dir", default="/app/output", help="Output directory for JSON files")
    
    args = parser.parse_args()
    
    print("🚀 Round 1A: Document Outline Extraction")
    print(f"📁 Input directory: {args.input_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Find all PDF files
    pdf_files = []
    for file in os.listdir(args.input_dir):
        if file.lower().endswith('.pdf'):
            pdf_files.append(file)
    
    if not pdf_files:
        print("❌ No PDF files found in input directory")
        sys.exit(1)
    
    print(f"📚 Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    start_time = time.time()
    success_count = 0
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(args.input_dir, pdf_file)
        output_file = pdf_file.replace('.pdf', '.json')
        output_path = os.path.join(args.output_dir, output_file)
        
        if process_pdf(pdf_path, output_path):
            success_count += 1
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n📊 Processing Summary:")
    print(f"   Total PDFs: {len(pdf_files)}")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {len(pdf_files) - success_count}")
    print(f"   Processing time: {processing_time:.2f} seconds")
    
    if processing_time > 10:
        print("⚠️  Warning: Processing time exceeds 10 seconds")
    else:
        print("✅ Processing time within 10-second limit")
    
    if success_count == len(pdf_files):
        print("🎉 All PDFs processed successfully!")
        return 0
    else:
        print("⚠️  Some PDFs failed to process")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 