# Challenge 1B: Persona-Driven Document Intelligence

## Overview

This solution addresses the "Connect What Matters" challenge by acting as an intelligent document analyst. It ingests a collection of PDF documents and, based on a specific user **persona** and their **job-to-be-done**, extracts, ranks, and analyzes the most relevant sections from across the entire document set. The solution is designed to be generic, capable of handling diverse document types and user roles, from academic research to business analysis.

### Core Features

* **Adaptive Section Extraction**: Intelligently parses PDFs with varied layouts by using a combination of heuristics to detect headings, ensuring robust structuring of content.
* **Hybrid Relevance Scoring**: A sophisticated ranking engine that combines the strengths of two methods for superior accuracy:
    * **Lexical Search (TF-IDF)**: Scores sections based on the frequency of important keywords.
    * **Semantic Search (Sentence Transformers)**: Goes beyond keywords to understand the meaning and context of the text, finding relevant information even if the exact words don't match.
* **Diversity-Aware Ranking**: A custom algorithm ensures the top-ranked results are not dominated by a single document, providing a comprehensive overview from the entire collection.
* **Fully Offline Execution**: All models (`all-MiniLM-L6-v2`) and data dependencies (NLTK) are pre-loaded and cached within the Docker image, allowing the container to run without any internet access, as per the hackathon rules.

---

## Technical Approach

The solution processes documents through a multi-stage pipeline designed for accuracy and efficiency.

1.  **Persona & Task Analysis**: The input `persona` and `job` descriptions are fed into the `ConfigurablePersonaAnalyzer`. This module uses the YAKE (Yet Another Keyword Extractor) library to distill the core concepts and creates a rich **"interest profile"** that serves as the master query for the system.

2.  **PDF Parsing & Structuring**: The `AdaptiveSectionExtractor` processes each PDF in the input collection. It uses the `PyMuPDF` library to extract text blocks and their properties (font, size). A rule-based system then identifies headings to logically segment the document content.

3.  **Hybrid Relevance Scoring**: The `ConfigurableRelevanceScorer` scores each extracted section against the interest profile. The final score is a weighted combination, $S_{final} = (w_{semantic} \cdot S_{semantic}) + (w_{tfidf} \cdot S_{tfidf})$, leveraging both semantic similarity from the `all-MiniLM-L6-v2` model and traditional TF-IDF keyword scores.

4.  **Ranking & Selection**: The sections are ranked by their final hybrid score. The `ensure_document_diversity` function then selects the top sections using a two-pass algorithm to guarantee the final output includes insights from multiple source documents.

5.  **Content Synthesis & Output**: For each of the top-ranked sections, the single most relevant paragraph is identified using semantic search. This forms the `refined_text` in the `subsection_analysis`. The final results, including metadata, are written to a structured JSON file.

---

## Offline Model Caching

[cite_start]To comply with the "no internet access" constraint[cite: 41], all required models are cached during the Docker build process when a network connection is available.

* A `preload_models.py` script is executed during the build.
* This script uses the `sentence-transformers` library to download the `all-MiniLM-L6-v2` model and saves it to a dedicated `/model_files` directory inside the image.
* The main application (`main_1b.py`) is configured to load the model directly from this local `/model_files` path, ensuring a fully offline operation at runtime.

---

## Build and Run Instructions

### Prerequisites
* Docker must be installed and the Docker daemon running.

### 1. Build the Docker Image
Navigate to this directory (`challenge_1b/`) and run the build command. This creates a self-contained image named `solution-1b`.

```bash
docker build -t solution-1b -f Dockerfile .
```

### 2. Run the Container
1.  Place your collection of PDFs into a subfolder inside this directory's `input/` folder (e.g., `input/my_pdfs/`).
2.  Run the command below from this directory (`challenge_1b/`). **Replace the values for `<...>`** to match your test case.

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  solution-1b \
  python main_1b.py \
    --input_dir "/app/input/<your_pdf_folder_name>" \
    --output_path "/app/output/results.json" \
    --persona "<Your Test Persona Description>" \
    --job "<The job your test persona needs to do>"
```

[cite_start]**Example (for Academic Research test case)[cite: 37]:**
```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none solution-1b python main_1b.py --input_dir "/app/input/research_papers" --output_path "/app/output/literature_review.json" --persona "PhD Researcher in Computational Biology" --job "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
```
The final analysis will be saved to `output/literature_review.json`.

---

## Constraints Compliance

This solution is designed to meet all specified constraints for Round 1B:

* [cite_start]**Runtime Environment**: Runs on CPU only and with no internet access during execution. [cite: 41]
* [cite_start]**Model Size**: The `all-MiniLM-L6-v2` model and its dependencies are well under the 1GB size limit. [cite: 40]
* [cite_start]**Processing Time**: The pipeline is optimized to complete its analysis of a 3-5 document collection within the 60-second time limit. [cite: 40]
