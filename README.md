Adobe India Hackathon 2025: Connecting the Dots

This repository contains the solutions for Round 1A and Round 1B of the Adobe India Hackathon 2025. The project is organized into separate directories for each challenge, each with its own self-contained, Dockerized environment.

Directory Structure

.
├── README.md                          # This file
│
├── challenge_1a/
│   ├── Dockerfile                     # Dockerfile for Challenge 1A
│   ├── README.md                      # Detailed README for Challenge 1A
│   ├── requirements.txt
│   ├── run_round1a.py                 # Script for Challenge 1A
│   ├── input/
│   └── output/
│
└── challenge_1b/
    ├── Dockerfile                     # Dockerfile for Challenge 1B
    ├── README.md                      # Detailed README for Challenge 1B
    ├── requirements.txt
    ├── preload_models.py
    ├── main_1b.py                     # Script for Challenge 1B
    ├── input/
    └── output/

Prerequisites

    Docker: You must have Docker installed and the Docker daemon running on your system.

Challenge 1A: Document Outline Extraction

This solution parses PDF files to extract a structured outline (Title, H1, H2, H3). For a detailed explanation of the methodology, please see the README.md file inside the challenge_1a directory.

How to Build

Bash

# Run this command from the project's root directory
docker build -t solution-1a -f ./challenge_1a/Dockerfile ./challenge_1a/

How to Run

    Place your input PDF files into the challenge_1a/input/ directory.

    Run the container from the project's root directory. The output JSONs will be generated in challenge_1a/output/.

Bash

# [cite_start]This command follows the execution specification [cite: 25]
docker run --rm \
  -v $(pwd)/challenge_1a/input:/app/input \
  -v $(pwd)/challenge_1a/output:/app/output \
  --network none \
  solution-1a

Challenge 1B: Persona-Driven Document Intelligence

This solution analyzes a collection of PDFs to extract and rank sections relevant to a given user persona and their job-to-be-done. For a detailed explanation, please see the README.md file inside the challenge_1b directory.

How to Build

Bash

# Run this command from the project's root directory
docker build -t solution-1b -f ./challenge_1b/Dockerfile ./challenge_1b/

How to Run

    Place your collection of PDFs into a subfolder inside challenge_1b/input/ (e.g., challenge_1b/input/my_collection/PDFs/).

    Run the command below from the project's root directory. Replace the <...> values for your specific test case.

Bash

docker run --rm \
  -v $(pwd)/challenge_1b/input:/app/input \
  -v $(pwd)/challenge_1b/output:/app/output \
  --network none \
  solution-1b \
  python main_1b.py \
    --input_dir "/app/input/<your_pdf_folder_name>" \
    --output_path "/app/output/results.json" \
    --persona "<Your Test Persona Description>" \
    --job "<The job your test persona needs to do>"

Example:
Bash

docker run --rm -v $(pwd)/challenge_1b/input:/app/input -v $(pwd)/challenge_1b/output:/app/output --network none solution-1b python main_1b.py --input_dir "/app/input/my_collection/PDFs" --output_path "/app/output/trip_plan.json" --persona "A travel planner for a group of 10 friends" --job "Plan a 4-day budget trip to the South of France."

