import os
import sys
import glob
import time
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# Load configurations from .env
load_dotenv()

OCR_PROVIDER = os.getenv("OCR_PROVIDER", "ocr_space").lower().strip()
MAX_SIZE_KB = int(os.getenv("MAX_IMAGE_SIZE_KB", 1000))
MAX_SIZE_BYTES = MAX_SIZE_KB * 1024

# Setup directories
INPUT_DIR = os.path.join("data", "book_1")
OUTPUT_DIR = os.path.join("data", "book_1_text")

# Ensure directories exist
if not os.path.exists(INPUT_DIR):
    print(f"[ERROR] Input directory '{INPUT_DIR}' does not exist.")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_api_credentials():
    """Validates and retrieves credentials based on selected provider."""
    if OCR_PROVIDER == "ocr_space":
        key = os.getenv("OCR_SPACE_KEY")
        if not key or key == "your_api_key_here":
            print("[ERROR] OCR_SPACE_KEY is not configured in .env file.")
            print("Please edit the '.env' file and place a valid key (e.g. 'helloworld' for testing).")
            sys.exit(1)
        url = "https://api.ocr.space/parse/image"
        return key, url
    elif OCR_PROVIDER == "api_ninjas":
        key = os.getenv("API_NINJAS_KEY")
        if not key or key == "your_api_key_here":
            print("[ERROR] API_NINJAS_KEY is not configured in .env file.")
            sys.exit(1)
        url = "https://api.api-ninjas.com/v1/imagetotext"
        return key, url
    else:
        print(f"[ERROR] Unknown OCR_PROVIDER '{OCR_PROVIDER}' in .env.")
        print("Please choose either 'ocr_space' or 'api_ninjas'.")
        sys.exit(1)


def compress_image_to_buffer(image_path, max_bytes):
    """
    Dynamically scales and compresses the image to fit under max_bytes,
    returning a bytes buffer.
    """
    img = Image.open(image_path)
    
    # Convert RGBA to RGB if necessary (JPEGs don't support alpha channel)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
        
    quality = 85
    width, height = img.size
    
    while True:
        # Save to buffer to check size
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        size = buf.tell()
        
        if size <= max_bytes:
            buf.seek(0)
            return buf, quality, img.size
            
        # If still too large, first try reducing quality
        if quality > 40:
            quality -= 10
        else:
            # If quality is already low (<= 40), scale down resolution by 15%
            width = int(width * 0.85)
            height = int(height * 0.85)
            if width < 100 or height < 100:
                buf.seek(0)
                return buf, quality, img.size
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            quality = 75  # Reset quality for smaller resolution


def format_api_ninjas_response(blocks):
    """Layout-aware stitching for API Ninjas OCR results."""
    if not blocks:
        return ""
        
    if not all(isinstance(b, dict) and "bounding_box" in b for b in blocks):
        return "\n".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in blocks])
        
    processed_blocks = []
    for b in blocks:
        text = b.get("text", "").strip()
        if not text:
            continue
        bbox = b.get("bounding_box", {})
        x = bbox.get("x", 0)
        y = bbox.get("y", 0)
        h = bbox.get("height", 10)
        processed_blocks.append({"text": text, "x": x, "y": y, "h": h})
        
    if not processed_blocks:
        return ""
        
    # Sort vertically
    processed_blocks.sort(key=lambda b: b["y"])
    
    lines = []
    current_line = []
    current_y_limit = -1
    
    for b in processed_blocks:
        if current_y_limit == -1:
            current_line.append(b)
            current_y_limit = b["y"] + (b["h"] * 0.6)
        elif b["y"] < current_y_limit:
            current_line.append(b)
        else:
            current_line.sort(key=lambda x: x["x"])
            lines.append(current_line)
            current_line = [b]
            current_y_limit = b["y"] + (b["h"] * 0.6)
            
    if current_line:
        current_line.sort(key=lambda x: x["x"])
        lines.append(current_line)
        
    line_texts = []
    for line in lines:
        line_texts.append(" ".join([b["text"] for b in line]))
        
    return "\n".join(line_texts)


def run_ocr_space(url, api_key, files):
    """Sends request to OCR.space API and parses the text."""
    payload = {
        "apikey": api_key,
        "language": "hin",  # Hindi covers Sanskrit/Devanagari scripts
        "isOverlayRequired": False,
        "OCREngine": 3      # Engine 3 is required for Hindi/Sanskrit
    }
    
    # OCR.space API expects a dict where value is (filename, file-like-object, mimetype)
    # or just the file-like-object
    response = requests.post(url, files=files, data=payload)
    
    if response.status_code != 200:
        print(f"  [ERROR] OCR.space API HTTP {response.status_code}: {response.text}")
        return None
        
    try:
        result = response.json()
    except Exception as e:
        print(f"  [ERROR] Failed to parse JSON response: {e}")
        return None
        
    if result.get("IsErroredOnProcessing"):
        error_msg = result.get("ErrorMessage", ["Unknown error occurred"])[0]
        print(f"  [ERROR] OCR.space Error: {error_msg}")
        return None
        
    parsed_results = result.get("ParsedResults", [])
    if not parsed_results:
        print("  [WARNING] No parsed results returned.")
        return ""
        
    return parsed_results[0].get("ParsedText", "").strip()


def run_api_ninjas(url, api_key, files):
    """Sends request to API Ninjas OCR API and parses the text."""
    headers = {"X-Api-Key": api_key}
    response = requests.post(url, files=files, headers=headers)
    
    if response.status_code != 200:
        print(f"  [ERROR] API Ninjas HTTP {response.status_code}: {response.text}")
        return None
        
    try:
        blocks = response.json()
        return format_api_ninjas_response(blocks)
    except Exception as e:
        print(f"  [ERROR] Failed to parse JSON response: {e}")
        return None


def process_images():
    api_key, api_url = get_api_credentials()
    
    # Find all image files
    img_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_paths = []
    for ext in img_extensions:
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
        
    # Sort files naturally (numerically/alphabetically)
    image_paths = sorted(list(set(image_paths)))
    
    total_images = len(image_paths)
    if total_images == 0:
        print(f"[WARNING] No image files found in '{INPUT_DIR}'.")
        return
        
    print(f"[INFO] Using OCR Provider: {OCR_PROVIDER}")
    print(f"[INFO] Found {total_images} images to process.")
    print(f"[INFO] Max upload size limit: {MAX_SIZE_KB} KB")
    
    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.basename(img_path)
        name_without_ext, _ = os.path.splitext(filename)
        output_txt_path = os.path.join(OUTPUT_DIR, f"{name_without_ext}.txt")
        
        # Check if already processed (supports resuming)
        if os.path.exists(output_txt_path):
            print(f"[{idx}/{total_images}] Skipping '{filename}' (already processed).")
            continue
            
        print(f"[{idx}/{total_images}] Processing '{filename}'...")
        
        original_size = os.path.getsize(img_path)
        
        # Compress if original file exceeds max size limit
        if original_size > MAX_SIZE_BYTES:
            print(f"  -> File size ({original_size / 1024:.1f} KB) exceeds limit. Compressing...")
            try:
                img_buffer, quality, new_dim = compress_image_to_buffer(img_path, MAX_SIZE_BYTES)
                compressed_size = len(img_buffer.getvalue())
                print(f"  -> Compressed to {compressed_size / 1024:.1f} KB (Quality: {quality}, Resized to: {new_dim[0]}x{new_dim[1]})")
                files = {"image": ("image.jpg", img_buffer, "image/jpeg")}
            except Exception as e:
                print(f"  [ERROR] Image compression failed: {e}")
                continue
        else:
            print(f"  -> Sending original file ({original_size / 1024:.1f} KB)")
            try:
                with open(img_path, "rb") as f:
                    # Create a buffer copy to be uniform
                    files = {"image": ("image.jpg", f.read(), "image/jpeg")}
            except Exception as e:
                print(f"  [ERROR] Failed to read file: {e}")
                continue
                
        # Perform OCR based on chosen provider
        extracted_text = None
        if OCR_PROVIDER == "ocr_space":
            extracted_text = run_ocr_space(api_url, api_key, files)
        elif OCR_PROVIDER == "api_ninjas":
            extracted_text = run_api_ninjas(api_url, api_key, files)
            
        if extracted_text is not None:
            # Write individual page text (encoded in UTF-8 for Devanagari script support)
            with open(output_txt_path, "w", encoding="utf-8") as out_f:
                out_f.write(extracted_text)
            print(f"  -> Saved text to {output_txt_path}")
        else:
            print(f"  [ERROR] OCR failed for '{filename}'")
            
        # Add a delay between requests to be good API citizens
        # OCR.space free API allows 1 request per few seconds, especially for 'helloworld'
        if OCR_PROVIDER == "ocr_space" and api_key == "helloworld":
            time.sleep(5)  # Longer delay for public trial key to avoid limits
        else:
            time.sleep(2)
            
    # Combine all processed pages together into a final output
    combine_pages(image_paths)


def combine_pages(image_paths):
    combined_path = os.path.join(OUTPUT_DIR, "combined_book.txt")
    print(f"[INFO] Combining all extracted text into '{combined_path}'...")
    
    with open(combined_path, "w", encoding="utf-8") as comb_f:
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            name_without_ext, _ = os.path.splitext(filename)
            txt_path = os.path.join(OUTPUT_DIR, f"{name_without_ext}.txt")
            
            if os.path.exists(txt_path):
                comb_f.write(f"\n--- PAGE: {name_without_ext} ({filename}) ---\n\n")
                with open(txt_path, "r", encoding="utf-8") as page_f:
                    comb_f.write(page_f.read())
                comb_f.write("\n")
                
    print("[SUCCESS] Processing and compilation complete!")


if __name__ == "__main__":
    process_images()
