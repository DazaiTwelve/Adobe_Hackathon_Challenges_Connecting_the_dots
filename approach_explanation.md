## **Round 1A: Document Outline Extraction**

Round 1A addresses the task of  **structural document analysis** , aiming to extract the title and hierarchical headings from PDF documents. The goal is to produce a navigable outline that reflects the logical organization of the content.

We use **PyMuPDF** for fast and reliable PDF parsing. The main component, `OutlineExtractor`, processes all text spans from the first 50 pages (as a performance safeguard), capturing metadata such as  **font size** ,  **bounding boxes** , and  **page numbers** .

The core logic is based on  **font size distribution analysis** . The most frequently occurring large font is identified as the  **title** , followed by  **H1** ,  **H2** , and **H3** headers in descending order. To improve robustness, we incorporate **regex-based detection** for numbered headings (e.g., "1.", "2.1") and **pattern-based heuristics** to recognize all-caps or title-case headings.

Each candidate heading is further validated using:

* **Text length constraints**
* **Capitalization and formatting checks**
* **Bounding box positioning**

The final output is a **clean JSON file** containing:

* A `title` field (document’s main title)
* An `outline` list with heading `level`, `text`, and `page`

This method ensures accurate structure extraction across a wide variety of documents with minimal noise or false positives.

---

## **Round 1B: Persona-Driven Content Intelligence**

Round 1B goes beyond structure, focusing on **semantic content extraction** driven by user **personas** and their  **job descriptions** . The objective is to prioritize sections that are most relevant to the user's information needs.

Our solution uses a  **three-stage pipeline** :

1. **Keyword Extraction** : We apply the **YAKE algorithm** to extract weighted keywords from persona and job role descriptions.
2. **Section Segmentation** : The document is segmented into sections using the same heading extraction engine as Round 1A (based on font size, layout cues, and patterns).
3. **Relevance Scoring** : Each section is evaluated using a  **hybrid scoring mechanism** :

* **Semantic similarity** via the compact **MiniLM (all-MiniLM-L6-v2)** model from Sentence Transformers
* **Lexical similarity** using **TF-IDF vectorization**

Final relevance scores are computed using a **60% semantic / 40% lexical** weighted combination, balancing contextual understanding with precise keyword matching.

Additional enhancements include:

* **Keyphrase highlighting**
* **Named entity recognition** (locations, activities)
* **Diversity filtering** to ensure coverage of distinct relevant topics

---

## **Constraints & Optimization**

Both rounds strictly adhere to hackathon constraints:

* **Offline execution**
* **CPU-only processing**
* **Model size under 1GB**
* **< 60 seconds processing time per document**

Together, these two methodologies provide a powerful combination—structural insight from Round 1A and persona-relevant content extraction from Round 1B—for comprehensive document intelligence.
