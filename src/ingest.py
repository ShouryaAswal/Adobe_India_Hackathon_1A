import json
import pymupdf
import pprint
import time
import os
from collections import Counter
import re
import pprint

def calculate_weighted_mean_font_size(blocks: list[dict]) -> float:
    """Calculates the character-weighted average font size for a list of blocks."""
    if not blocks:
        return 0.0

    total_char_length = sum(len(block['text']) for block in blocks)
    if total_char_length == 0:
        return 0.0

    weighted_font_sum = sum(block['font_size'] * len(block['text']) for block in blocks)
    return weighted_font_sum / total_char_length

def filter_long_blocks(blocks: list[dict], weighted_avg_font_size: float, max_words: int = 20) -> list[dict]:
    """
    Filters out text blocks that exceed a specified word count, unless they
    are significantly larger than the average font size (likely a title).
    """
    if not blocks:
        return []
        
    final_blocks = []
    for block in blocks:
        is_long = len(block['text'].split()) > max_words
        is_large_font = block['font_size'] > (weighted_avg_font_size * 1.5)

        # Keep the block if it's not long, OR if it's long but has a very large font.
        if not is_long or is_large_font:
            final_blocks.append(block)
            
    return final_blocks

def filter_small_fonts_by_weighted_mean(blocks: list[dict], weighted_avg_font_size: float) -> list[dict]:
    """
    Filters out blocks with font sizes at or below a character-weighted average,
    unless the block has other heading-like features like boldness or a unique color.
    """
    if not blocks:
        return []

    # Find the majority color in the document
    colors = [block['color'] for block in blocks]
    majority_color = Counter(colors).most_common(1)[0][0] if colors else 0
    print(f"Found majority text color: {majority_color}")
    
    # --- Reverted to single-pass logic ---
    # Keep blocks if they are larger than the average, OR if they are bold,
    # OR if their color is not the majority color.
    final_blocks = [
        block for block in blocks 
        if block['font_size'] > weighted_avg_font_size or 
           block['is_bold'] or 
           block['color'] != majority_color
    ]

    return final_blocks

def find_repeating_texts(blocks: list[dict], page_count: int, min_occurrence_ratio: float = 0.5) -> set:
    """
    Finds text content that repeats across a significant number of pages.
    """
    if not blocks or page_count == 0:
        return set()

    text_page_map = {}
    for block in blocks:
        text = block['text']
        page = block['page_number']
        if text not in text_page_map:
            text_page_map[text] = set()
        text_page_map[text].add(page)
    
    min_pages = int(page_count * min_occurrence_ratio)
    if min_pages < 2: min_pages = 2

    repeating_texts = {text for text, pages in text_page_map.items() if len(pages) >= min_pages}
    
    return repeating_texts

def filter_header_footer_blocks(blocks: list[dict], page_count: int, header_margin: float = 0.12, footer_margin: float = 0.12) -> list[dict]:
    """
    Filters out blocks that are likely headers or footers based on repetition and position.
    """
    if not blocks:
        return []

    repeating_texts = find_repeating_texts(blocks, page_count)
    if not repeating_texts:
        return blocks

    page_height = 792 
    header_threshold = page_height * header_margin
    footer_threshold = page_height * (1 - footer_margin)

    filtered_blocks = []
    for block in blocks:
        is_repeating = block['text'] in repeating_texts
        is_in_header = block['bbox'][1] < header_threshold
        is_in_footer = block['bbox'][3] > footer_threshold

        if not (is_repeating and (is_in_header or is_in_footer)):
            filtered_blocks.append(block)
            
    return filtered_blocks

def post_process_blocks(blocks: list[dict]) -> list[dict]:
    """
    Cleans and finalizes the text content of each logical block.
    """
    processed_blocks = []
    for block in blocks:
        text = block["text"].strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^\W+|\W+$', '', text)
        
        # --- New: Only keep the block if it contains at least one alphabetic character ---
        if text and re.search('[a-zA-Z]', text):
            block["text"] = text
            processed_blocks.append(block)
            
    return processed_blocks

def extract_logical_text_blocks(pdf_path: str, line_proximity_threshold: float = 4.0) -> tuple[list[dict], int]:
    """
    Extracts logically coherent text blocks by merging lines based on style and proximity.
    """
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return [], 0

    final_blocks = []
    page_count = doc.page_count
    for page_num, page in enumerate(doc):
        flags = pymupdf.TEXT_PRESERVE_LIGATURES | pymupdf.TEXT_DEHYPHENATE
        page_dict = page.get_text("dict", flags=flags)
        
        spans = page_dict["blocks"]
        lines = []
        for block in spans:
            if block.get("type") == 0 and "lines" in block:
                for line in block["lines"]:
                    line_text, style_counter, line_bbox = "", Counter(), pymupdf.Rect()
                    for span in line["spans"]:
                        font_name_lower = span.get("font", "").lower()
                        is_bold_by_flag = (span.get("flags", 0) & 2**4) > 0
                        is_bold_by_name = "bold" in font_name_lower or "heavy" in font_name_lower
                        is_bold = is_bold_by_flag or is_bold_by_name
                        
                        # Add color to the style tuple
                        color = span.get("color", 0)
                        span_style = (round(span["size"]), span["font"], is_bold, color)
                        
                        span_text = span["text"]
                        style_counter[span_style] += len(span_text)
                        line_text += span_text
                        line_bbox.include_rect(span["bbox"])
                    if line_text.strip():
                        if not style_counter: continue
                        dominant_style = style_counter.most_common(1)[0][0]
                        lines.append({"text": line_text, "bbox": line_bbox, "style": dominant_style})
        
        if not lines: continue

        merged_blocks = []
        if lines:
            current_block = {"text": lines[0]["text"], "bbox": pymupdf.Rect(lines[0]["bbox"]), "style": lines[0]["style"]}
            for i in range(1, len(lines)):
                prev_line, current_line = lines[i-1], lines[i]
                same_style = (current_line["style"] == prev_line["style"])
                vertically_close = (current_line["bbox"].y0 - prev_line["bbox"].y1) < line_proximity_threshold
                is_list_item = re.match(r'^\s*([•-]|(\d+\.))\s+', current_line['text'])
                if same_style and vertically_close and not is_list_item:
                    current_block["text"] += " " + current_line["text"]
                    current_block["bbox"].include_rect(current_line["bbox"])
                else:
                    merged_blocks.append(current_block)
                    current_block = {"text": current_line["text"], "bbox": pymupdf.Rect(current_line["bbox"]), "style": current_line["style"]}
            merged_blocks.append(current_block)
        
        for block in merged_blocks:
            final_blocks.append({
                "page_number": page_num + 1,
                "text": block["text"],
                "bbox": tuple(block["bbox"]),
                "font_size": block["style"][0],
                "font_name": block["style"][1],
                "is_bold": block["style"][2],
                "color": block["style"][3]
            })

    doc.close()
    return final_blocks, page_count

def engineer_layout_features(blocks: list[dict]) -> list[dict]:
    """This function remains the same and works with the new block structure."""
    if not blocks: return []
    font_sizes = [block['font_size'] for block in blocks if block['font_size'] > 0]
    if not font_sizes: return []
    modal_font_size = Counter(font_sizes).most_common(1)[0][0]
    for block in blocks:
        block['relative_font_size'] = block['font_size'] / modal_font_size if modal_font_size > 0 else 0
        block['is_bold_numeric'] = 1 if block['is_bold'] else 0
        block['char_count'] = len(block['text'])
        block['word_count'] = len(block['text'].split())
        page_height = 792 
        block['vertical_position'] = block['bbox'][1] / page_height if page_height > 0 else 0
    return blocks

# --- Verification Step ---
if __name__ == "__main__":
    sample_dir = "sample_dataset/input"
    sample_pdf_path = os.path.join(sample_dir, "file02.pdf")
    
    if not os.path.exists(sample_pdf_path):
        print(f"'{sample_pdf_path}' not found. Creating a dummy PDF for testing.")
        os.makedirs(sample_dir, exist_ok=True)
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 72), "This is a Main Title With More Than Twenty Words To Test The New Logic", fontsize=18, fontname="helv-bold")
        page.insert_text((50, 100), "This is a short subtitle", fontsize=14, fontname="helv-bold")
        page.insert_text((50, 130), "This is a very long block of text that should definitely be removed because it has way more than twenty words in it, which is the cutoff.", fontsize=11, fontname="helv")
        page.insert_text((50, 160), "Another short heading", fontsize=11, fontname="helv-bold")
        page.insert_text((50, 190), "This is a paragraph that will be deleted, confirming the block above is a heading.", fontsize=11, fontname="helv")
        page.insert_text((50, 750), "Confidential Footer", fontsize=9, fontname="helv")
        doc.save(sample_pdf_path)
        doc.close()
        print("Dummy PDF created.")

    start_time = time.monotonic()

    print(f"\nProcessing '{sample_pdf_path}'...")
    extracted_blocks, page_count = extract_logical_text_blocks(sample_pdf_path)
    print(f"1. Extraction complete. Found {len(extracted_blocks)} logical blocks.")
    
    non_header_footer_blocks = filter_header_footer_blocks(extracted_blocks, page_count)
    print(f"2. Header/Footer filtering complete. {len(non_header_footer_blocks)} blocks remaining.")

    # --- New Order: Calculate font size threshold before other filters ---
    weighted_font_threshold = calculate_weighted_mean_font_size(non_header_footer_blocks)
    print(f"Calculated weighted average font size threshold: {weighted_font_threshold:.2f}")

    short_blocks = filter_long_blocks(non_header_footer_blocks, weighted_font_threshold)
    print(f"3. Long block filtering complete. {len(short_blocks)} blocks remaining.")

    font_filtered_blocks = filter_small_fonts_by_weighted_mean(short_blocks, weighted_font_threshold)
    print(f"4. Small font filtering complete. {len(font_filtered_blocks)} blocks remaining.")

    final_blocks = post_process_blocks(font_filtered_blocks)
    print(f"5. Post-processing and cleaning complete. {len(final_blocks)} blocks remaining.")

    featured_blocks = engineer_layout_features(final_blocks)
    print(f"6. Successfully engineered layout features.")

    end_time = time.monotonic()
    duration = end_time - start_time
    print(f"\n--- Total processing time: {duration:.4f} seconds ---")

    print("\n--- Verification: Text Content of Final Blocks (Post-Filtering) ---")
    if featured_blocks:
        for i, block in enumerate(featured_blocks):
            # --- Updated: Print color along with size ---
            print(f"Block {i+1} (Page {block['page_number']}): {block['text']} [Size: {block['font_size']}, Color: {block['color']}]")
            print("-" * 20)
        # for i, block in enumerate(featured_blocks):
        #     # --- Updated: Print the entire feature dictionary for each block ---
        #     print(f"--- Block {i+1} ---")
        #     pprint.pprint(block)
    else:
        print("No blocks remained after filtering.")

    # --- Save the final blocks to a JSON file ---
    if featured_blocks:
        # Define the output path for the JSON file
        output_dir = "sample_dataset/text-blocks"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(sample_pdf_path))[0]
        output_json_path = os.path.join(output_dir, f"{base_name}_blocks.json")

        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(featured_blocks, f, ensure_ascii=False, indent=4)
            print(f"\nSuccessfully saved {len(featured_blocks)} blocks to '{output_json_path}'.")
        except Exception as e:
            print(f"\nError saving JSON file: {e}")
