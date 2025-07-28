# AI-Powered PDF Structure Extraction (Adobe Hackathon - Round 1A)

This repository contains the complete solution for Round 1A of the Adobe India Hackathon. The project implements a robust, machine learning-based pipeline to perform hierarchical structure extraction (Title, H1, H2, H3) from PDF documents. The system is designed for high accuracy and performance, adhering to all specified constraints, including offline execution, CPU-only inference, and strict resource limits.

## 1. How to Run the Solution

This solution is delivered as a self-contained Docker image. The following instructions detail how to build the image and run the inference pipeline on a collection of PDF documents.

### Prerequisites
* Docker Desktop installed and running.
* A local folder containing the input PDF files.

### Step 1: Build the Docker Image

Navigate to the root directory of this repository in your terminal. Execute the following command to build the Docker image. This command reads the `Dockerfile`, installs all dependencies, and copies the source code and the fine-tuned model into the image.

```bash
docker build --platform linux/amd64 -t adobe-1a-solution .
```

### Step 2: Run the Inference Container

To process your PDF files, use the `docker run` command below. This command mounts your local input and output directories into the container, allowing the script to process the files and save the results back to your local machine.

* **For PowerShell (Windows):**
    ```powershell
    docker run --rm -v "${PWD}/path/to/your/input:/app/input" -v "${PWD}/path/to/your/output:/app/output" --network none adobe-1a-solution
    ```

* **For Bash (Linux/macOS):**
    ```bash
    docker run --rm -v "$(pwd)/path/to/your/input:/app/input" -v "$(pwd)/path/to/your/output:/app/output" --network none adobe-1a-solution
    ```
**Note:** Replace `path/to/your/input` and `path/to/your/output` with the actual relative paths to your data folders.

The container will automatically process every `.pdf` file in the mounted input directory and generate a corresponding `.json` file in the output directory.

## 2. The Development Process: From Data to Deployment

The success of this project hinges on a meticulous, multi-stage development process that treats heading detection not as a simple rule-based problem, but as a nuanced sequence classification task.

### 2.1. Feature Engineering (`ingest.py`)

The foundational step was to design a feature extraction pipeline that moves beyond naive font-size thresholds. Our `ingest.py` script uses `PyMuPDF` to parse each document and constructs a rich feature vector for every text block. These features include:

* **Stylistic:** Font name, font size, and a boolean `is_bold` flag.
* **Positional:** The normalized vertical position (`vertical_position`) of the block on the page.
* **Content-based:** The total word count (`word_count`) of the block.
* **Relational:** The block's font size relative to the document's most common font size (`relative_font_size`). This crucial feature helps normalize layouts across different documents.

### 2.2. Data Curation (`create_training_data.py` & `generate_heuristic_data.py`)

A high-quality model requires high-quality data. We bootstrapped our training dataset through a semi-automated process:

1.  **Heuristic Labeling:** `generate_heuristic_data.py` was used to apply a set of intelligent rules to an initial corpus of documents, creating a baseline labeled dataset.
2.  **Manual Curation:** This baseline was then manually reviewed and corrected to ensure high-fidelity ground truth labels (`Title`, `H1`, `H2`, `H3`, `P`).
3.  **Semantic Tagging:** The core of our feature representation is handled by `create_training_data.py`. This script converts the raw numerical features into a set of discrete **semantic tags** (e.g., `[FONT_HIGH]`, `[POS_TOP]`, `[STYLE_BOLD]`). This abstraction allows the model to learn from categorical descriptions rather than overfitting to specific numerical values. The final input to the model is a "feature string" that combines these tags with the raw text, like so:
    `[STYLE_BOLD] [FONT_HIGH] [POS_TOP] ... The actual text of the block`

### 2.3. Dataset Balancing (`balance_dataset.py`)

Document structures are inherently imbalanced (many paragraphs, few titles). `balance_dataset.py` was employed to perform down-sampling on the majority class (`P`), preventing the model from developing a bias towards simply classifying everything as a paragraph and ensuring it paid sufficient attention to the critical heading classes.

### 2.4. Model Selection and Training (`training.py`)

The selection of a model was governed by a strict set of constraints: under 200MB, high performance on CPU, and strong multilingual capabilities for future-proofing.

#### **Initial Approaches and Challenges:**

* **Quantized Models:** We initially explored using standard models like `bert-base-uncased` and applying post-training dynamic quantization. While this significantly reduced model size, we observed an unacceptable degradation in classification accuracy, particularly for nuanced heading levels like H2 vs. H3. The precision-recall trade-off was too steep.
* **SqueezeBERT:** We also experimented with `SqueezeBERT`, a model architecture designed for on-device performance. However, its performance on our specific feature-rich classification task did not meet our accuracy targets. It appeared to struggle with the non-natural language component of our feature strings.

#### **The Final Choice: `dbmdz/bert-mini-historic-multilingual-cased`**

We ultimately selected a fine-tuned version of `dbmdz/bert-mini-historic-multilingual-cased`. This model provided the optimal balance of all our requirements:
* **Size:** The fine-tuned model comes in well under the 200MB limit.
* **Performance:** It demonstrated high accuracy on our validation set, effectively learning the patterns from our semantic tags.
* **Efficiency:** As a smaller BERT variant, its inference speed on CPU is excellent and well within the 10-second requirement.
* **Multilingual:** Its multilingual pre-training provides a significant advantage and aligns with the bonus criteria of the hackathon.

The `training.py` script orchestrated the fine-tuning process using the `transformers` library on our curated, balanced dataset.

## 3. System Architecture and Final Pipeline

The final solution is a streamlined, end-to-end pipeline encapsulated within a Docker container.

1.  **`Dockerfile`:** Defines the environment, starting from a `python:3.10-slim` base image, installing all dependencies from `requirements.txt`, and copying the `src` and `model` directories.
2.  **`src/main.py`:** The entry point of the container. It loads the fine-tuned model and tokenizer from `/app/model` **once** at startup. It then iterates through every PDF in the `/app/input` directory, executing the full feature engineering and classification pipeline for each, and saves the structured JSON to `/app/output`.
3.  **`src/ingest.py`:** Contains the core, reusable functions for PDF parsing and feature engineering, called by `main.py`.

This architecture ensures that the solution is isolated, reproducible, and highly efficient, meeting all technical specifications of the challenge.
