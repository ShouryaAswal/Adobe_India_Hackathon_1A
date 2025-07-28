# src/main.py
import os
import json
import torch
import time
import pandas as pd
import numpy as np
from pathlib import Path

# --- Make sure this import works by having an __init__.py file in the src folder ---
from ingest import extract_logical_text_blocks, engineer_layout_features

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.pipelines import pipeline

# ==============================================================================
# --- Docker-Compliant Configuration ---
# ==============================================================================
INPUT_DIR = Path("/app/input")
OUTPUT_DIR = Path("/app/output")
MODEL_DIR = Path("/app/model")

# ==============================================================================
# --- Semantic Tagging Logic (Integrated from create_training_data.py) ---
# ==============================================================================
def create_semantic_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw numerical features into discrete semantic tags.
    This logic MUST be identical to the logic used for training.
    """
    if df.empty:
        return df

    # --- Rule-Based Tagging for Normalized/Predictable Features ---
    # Rule for 'relative_font_size'
    rfs_conditions = [df['relative_font_size'] > 1.15, df['relative_font_size'] < 0.95]
    rfs_choices = ['[RFS_HIGH]', '[RFS_LOW]']
    df['tag_rel_font'] = np.select(rfs_conditions, rfs_choices, default='[RFS_MEDIUM]')

    # Rule for 'vertical_position'
    pos_conditions = [df['vertical_position'] < 0.25, df['vertical_position'] > 0.75]
    pos_choices = ['[POS_TOP]', '[POS_BOTTOM]']
    df['tag_v_pos'] = np.select(pos_conditions, pos_choices, default='[POS_MIDDLE]')

    # Rule for 'word_count'
    wc_conditions = [df['word_count'] > 15, df['word_count'] < 5]
    wc_choices = ['[WORDS_HIGH]', '[WORDS_LOW]']
    df['tag_words'] = np.select(wc_conditions, wc_choices, default='[WORDS_MEDIUM]')

    # --- Statistical Binning for features without a predictable scale (like absolute font_size) ---
    try:
        df['tag_font_size'] = pd.qcut(df['font_size'], q=3, labels=['[FONT_LOW]', '[FONT_MEDIUM]', '[FONT_HIGH]'], duplicates='drop')
    except ValueError:
        # Fallback for when all font sizes are the same (cannot create bins)
        df['tag_font_size'] = '[FONT_MEDIUM]'

    # Safely handle the boolean 'is_bold' feature
    if 'is_bold' in df.columns:
        df['tag_bold'] = df['is_bold'].apply(lambda x: '[STYLE_BOLD]' if x else '[STYLE_NOT_BOLD]')
    else:
        df['tag_bold'] = '[STYLE_NOT_BOLD]'

    return df

# ==============================================================================
# --- Feature String Creation (from training.py) ---
# ==============================================================================
def create_feature_string_from_tags(record: dict) -> str:
    """
    Converts a record with semantic tags into a single string for the model.
    This must match the format used during training.
    """
    text = record.get('text', '')
    tags = [
        record.get('tag_bold'),
        record.get('tag_font_size'),
        record.get('tag_rel_font'),
        record.get('tag_v_pos'),
        record.get('tag_words'),
    ]
    feature_str = " ".join(tag for tag in tags if tag)
    return f"{feature_str} {text}"

# ==============================================================================
# --- Main Inference Pipeline ---
# ==============================================================================
def process_single_pdf(pdf_path: Path, classifier):
    """Processes a single PDF, from feature extraction to prediction."""
    print(f"  -> Processing: {pdf_path.name}")
    
    # 1. Extract raw blocks and engineer numerical features
    try:
        raw_blocks, _ = extract_logical_text_blocks(str(pdf_path))
        if not raw_blocks:
            print(f"  Warning: No text blocks extracted from {pdf_path.name}.")
            return None
        featured_blocks = engineer_layout_features(raw_blocks)
    except Exception as e:
        print(f"  Error during PDF processing with 'ingest' functions: {e}")
        return None

    # 2. Convert numerical features to semantic tags
    featured_df = pd.DataFrame(featured_blocks)
    tagged_df = create_semantic_tags(featured_df)
    tagged_blocks = tagged_df.to_dict('records')

    # 3. Prepare tagged blocks for inference
    inference_texts = [create_feature_string_from_tags(block) for block in tagged_blocks]

    # 4. Run prediction
    try:
        predictions = classifier(inference_texts, top_k=1, truncation=True, padding=True)
    except Exception as e:
        print(f"  Error during model inference: {e}")
        return None

    # 5. Assemble final JSON output
    outline = []
    title_text = pdf_path.stem # Default title

    for i, block in enumerate(tagged_blocks):
        predicted_label = predictions[i][0]['label'] if predictions and i < len(predictions) and predictions[i] else "P"
        
        if predicted_label in ["Title", "H1", "H2", "H3"]:
            outline.append({
                "level": predicted_label,
                "text": block.get("text", ""),
                "page": block.get("page_number", 0)
            })
    
    # Logic to find the best title from the predictions
    if outline:
        title_items = [item for item in outline if item['level'] == 'Title']
        if title_items:
            title_text = title_items[0]['text']
        else:
            h1_items = [item for item in outline if item['level'] == 'H1']
            if h1_items:
                title_text = h1_items[0]['text']

    # Filter the outline to only include H1, H2, H3 (Title is separate)
    final_outline = [item for item in outline if item['level'] != 'Title']

    return {"title": title_text, "outline": final_outline}


if __name__ == "__main__":
    print("--- Starting Document Outline Extraction Pipeline ---")

    # 1. Load the fine-tuned model and tokenizer ONCE
    print("Loading classification model...")
    if not MODEL_DIR.is_dir():
        print(f"FATAL: Model directory not found at {MODEL_DIR}. Exiting.")
        exit()

    try:
        classifier = pipeline(
            "text-classification",
            model=str(MODEL_DIR),
            tokenizer=str(MODEL_DIR),
            device=-1 # Forcing CPU as per hackathon constraints
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not load model. Error: {e}")
        exit()

    # 2. Process all PDF files found in the input directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF(s) to process in {INPUT_DIR}.")

    for pdf_path in pdf_files:
        start_time = time.time()
        output_data = process_single_pdf(pdf_path, classifier)
        
        if output_data is None:
            print(f"  -> Skipping {pdf_path.name} due to a processing error.")
            continue
            
        json_filename = f"{pdf_path.stem}.json"
        output_path = OUTPUT_DIR / json_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        
        duration = time.time() - start_time
        print(f"  -> Finished {pdf_path.name} in {duration:.2f}s. Output saved to {output_path}.")
            
    print("--- All files processed. Pipeline finished. ---")