# Challenge 1A: Document Outline Extraction

## Overview

This solution addresses the "Understand Your Document" challenge by parsing a given PDF file and extracting its structural outline. The primary mission is to process a raw PDF and produce a clean, hierarchical outline containing the document's title and all H1, H2, and H3 level headings. This structured data serves as a foundational layer for more advanced document intelligence applications.

### Core Features

* **Accepts PDF Files**: The solution is containerized with Docker and processes all PDF files placed in a designated input directory.
* **Hierarchical Outline Extraction**: It identifies and extracts the document title and headings (H1, H2, H3) along with their respective levels and page numbers.
* **Structured JSON Output**: The final output is a valid JSON file that strictly adheres to the format specified in the challenge guidelines.
* **Offline Execution**: The Docker container runs entirely without network access, ensuring compliance with all challenge constraints.

---

## Technical Approach

The solution employs a hybrid approach, combining font-size analysis with pattern matching to identify structural elements within the PDF, making it robust against documents with inconsistent formatting.

1.  **PDF Parsing**: The system ingests the PDF using the `PyMuPDF` library. This allows for the extraction of not just raw text, but also critical metadata for each text span, including its font size and page number.

2.  **Font-Size Analysis**: The script analyzes the frequency of all font sizes in the document. It assumes that the largest, less frequent font sizes correspond to the Title and H1-H3 headings.

3.  **Rule-Based Validation**: To improve accuracy and address the pro-tip that "font size is not always reliable", the text identified by font analysis is further validated against a set of regular expressions and heuristics (`_is_valid_heading`). This checks for common heading patterns (e.g., numbered lists, title case) and filters out non-heading text.

4.  **Hierarchy Construction**: The system constructs the final JSON structure from the validated list of headings, ensuring the output is clean and correctly formatted.

---

## Build and Run Instructions

### Prerequisites
* Docker must be installed and the Docker daemon running.

### 1. Build the Docker Image
Navigate to this directory (`challenge_1a/`) and run the following command to build the image.

```bash
docker build -t solution-1a -f Dockerfile .
```

### 2. Run the Container
1.  Place your input PDF files into the `input/` directory within this folder.
2.  Run the container using the command below. The script will automatically process all PDFs.
3.  The corresponding JSON output files will be generated in the `output/` directory.

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  solution-1a
```

---

## Constraints Compliance

This solution is designed to meet all specified constraints for Round 1A:

* **Execution Time**: Processes a 50-page PDF in under 10 seconds.
* **Model Size**: Does not use an ML model, staying well under the 200MB limit.
* **Runtime Environment**: Runs on `linux/amd64` CPU architecture with no GPU dependencies and no internet calls at runtime.
