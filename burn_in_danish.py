import os, tempfile, shutil, json, base64, re
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import requests

from deep_translator import GoogleTranslator
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip

# OpenRouter API key — set via environment variable
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


# -----------------------------
# Video helpers
# -----------------------------
def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = frames / fps if frames > 0 else None
    cap.release()
    return {"fps": float(fps), "w": w, "h": h, "frames": frames, "duration": dur}


def read_frame_at_time(path, t_sec):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = int(round(t_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame at t={t_sec:.2f}s")
    return frame  # BGR


def frame_to_base64(frame_bgr):
    """Convert BGR frame to base64-encoded JPEG."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    import io
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# -----------------------------
# Vision API for text extraction
# -----------------------------
def extract_text_with_vision(frame_bgr, frame_time):
    """
    Use OpenRouter vision model to extract ALL text from frame.
    Returns list of text blocks with approximate positions.
    """
    base64_img = frame_to_base64(frame_bgr)
    h, w = frame_bgr.shape[:2]
    
    prompt = """You are analyzing a video frame to extract TEXT OVERLAYS/CAPTIONS that were intentionally added to the video.

FOCUS ON: Main title text, subtitles, captions, or informational text overlays that appear prominently on the video.
IGNORE: Background text in the scene (signs, logos on objects, watermarks, small incidental text, numbers on cars/objects).

For each INTENTIONAL text overlay, provide:
1. The complete text (preserve newlines if multi-line)
2. Position: "top", "center", or "bottom" of the frame
3. Color of the text (e.g., "white", "yellow")
4. Has stroke/outline: true or false
5. Size: "small", "medium", or "large"

Return JSON array format:
[{"text": "full text here", "position": "bottom", "color": "white", "has_stroke": true, "size": "large"}]

If no intentional caption/overlay text exists, return: []

IMPORTANT: Only include text that was ADDED as a caption/overlay, not text that exists in the physical scene being filmed."""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "google/gemini-2.0-flash-001",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                        ]
                    }
                ],
                "max_tokens": 1000
            },
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"  Vision API error: {response.status_code} - {response.text[:200]}")
            return []
        
        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Parse JSON from response
        json_match = re.search(r'\[[\s\S]*?\]', content)
        if json_match:
            text_blocks = json.loads(json_match.group())
            return text_blocks
        
        return []
        
    except Exception as e:
        print(f"  Vision API exception: {e}")
        return []


def normalize_text_for_comparison(text):
    """Normalize text for comparison - lowercase, remove extra whitespace."""
    import re
    text = re.sub(r'\s+', ' ', text.strip().lower())
    text = re.sub(r'[^a-z0-9 ]', '', text)  # Remove punctuation
    return text


def get_combined_text(blocks):
    """Get all text from blocks combined into one string."""
    if not blocks:
        return ""
    texts = [b.get("text", "").strip() for b in blocks if b.get("text")]
    return ' '.join(texts)


def texts_are_similar(text1, text2):
    """Check if two texts are similar (one contains the other or high overlap)."""
    n1 = normalize_text_for_comparison(text1)
    n2 = normalize_text_for_comparison(text2)
    
    if not n1 or not n2:
        return n1 == n2
    
    # Check if one contains the other
    if n1 in n2 or n2 in n1:
        return True
    
    # Check word overlap
    words1 = set(n1.split())
    words2 = set(n2.split())
    if not words1 or not words2:
        return False
    
    overlap = len(words1 & words2) / max(len(words1), len(words2))
    return overlap >= 0.6  # 60% word overlap


# -----------------------------
# Video text segment detection
# -----------------------------
def detect_text_segments_from_video(video_path, sample_interval=1.0):
    """
    Sample frames and use vision model to detect all text.
    Group consecutive frames with similar text into segments.
    """
    info = video_info(video_path)
    duration = info["duration"]
    h, w = info["h"], info["w"]
    
    print(f"Video: {duration:.2f}s at {info['fps']:.1f} fps, {w}x{h}")
    
    raw_detections = []
    t = 0.0
    
    while t < duration:
        print(f"  Analyzing t={t:.1f}s...", end=" ", flush=True)
        try:
            frame = read_frame_at_time(video_path, t)
            blocks = extract_text_with_vision(frame, t)
            
            if blocks:
                print(f"found {len(blocks)} text block(s)")
                for b in blocks:
                    print(f"    - {b.get('position', '?')}: \"{b.get('text', '')[:50]}...\"" if len(b.get('text', '')) > 50 else f"    - {b.get('position', '?')}: \"{b.get('text', '')}\"")
            else:
                print("no text")
            
            raw_detections.append({
                "time": t,
                "blocks": blocks,
                "combined_text": get_combined_text(blocks),
                "frame_size": (w, h)
            })
            
        except Exception as e:
            print(f"error: {e}")
            raw_detections.append({"time": t, "blocks": [], "combined_text": "", "frame_size": (w, h)})
        
        t += sample_interval
    
    # Group into segments - merge similar consecutive text
    if not raw_detections:
        return []
    
    segments = []
    current_start = None
    current_blocks = None
    current_text = ""
    
    for det in raw_detections:
        det_text = det["combined_text"]
        
        # Check if this detection is similar to current segment
        is_similar = texts_are_similar(current_text, det_text) if current_text and det_text else False
        is_continuation = is_similar or (not det_text and current_text)  # Allow gaps if no text detected
        
        if not is_continuation:
            # End previous segment
            if current_text and current_blocks:
                segments.append({
                    "start": current_start,
                    "end": det["time"],
                    "blocks": current_blocks,
                    "frame_size": det["frame_size"]
                })
            
            # Start new segment
            current_start = det["time"] if det_text else None
            current_blocks = det["blocks"] if det["blocks"] else None
            current_text = det_text
        else:
            # Continue current segment, but keep the longest/most complete text
            if len(det_text) > len(current_text) and det["blocks"]:
                current_blocks = det["blocks"]
                current_text = det_text
    
    # End final segment
    if current_text and current_blocks:
        segments.append({
            "start": current_start,
            "end": duration,
            "blocks": current_blocks,
            "frame_size": raw_detections[-1]["frame_size"]
        })
    
    return segments


# -----------------------------
# Translation with LLM for natural Danish
# -----------------------------
def translate_to_danish_natural(text):
    """Use OpenRouter to translate text to natural, idiomatic Danish."""
    prompt = f"""Translate the following English text to natural, idiomatic Danish.

IMPORTANT:
- Do NOT translate word-for-word literally
- The translation should sound like a native Danish speaker wrote it
- Keep the same meaning and tone but use natural Danish phrasing
- Keep it concise for video subtitles

English text:
"{text}"

Respond with ONLY the Danish translation, nothing else."""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "google/gemini-2.0-flash-001",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500
            },
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"  Translation API error: {response.status_code}")
            return text
        
        result = response.json()
        translated = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return translated if translated else text
        
    except Exception as e:
        print(f"  Translation exception: {e}")
        return text


def translate_segments_to_danish(segments):
    """Translate all text blocks in segments to natural Danish."""
    for seg in segments:
        for block in seg.get("blocks", []):
            text = block.get("text", "")
            if text:
                block["text_da"] = translate_to_danish_natural(text)
                print(f"  '{text[:40]}...' -> '{block['text_da'][:40]}...'" if len(text) > 40 else f"  '{text}' -> '{block['text_da']}'")
    
    return segments


# -----------------------------
# Style and rendering
# -----------------------------
def position_to_bbox(position, frame_w, frame_h, block_height=100):
    """Convert position string to approximate bbox coordinates."""
    margin_x = int(frame_w * 0.05)
    
    if position == "top":
        y1 = int(frame_h * 0.08)
    elif position == "bottom":
        y1 = int(frame_h * 0.75)
    else:  # center
        y1 = int(frame_h * 0.45)
    
    return (margin_x, y1, frame_w - margin_x, y1 + block_height)


def color_name_to_rgb(color_name):
    """Convert color name to RGB tuple."""
    colors = {
        "white": (255, 255, 255),
        "yellow": (255, 255, 0),
        "red": (255, 50, 50),
        "blue": (50, 100, 255),
        "green": (50, 255, 50),
        "black": (0, 0, 0),
        "orange": (255, 165, 0),
        "pink": (255, 105, 180),
    }
    return colors.get(color_name.lower(), (255, 255, 255))


def pick_best_font(fonts_dir):
    """Return the first available bold font."""
    if not os.path.isdir(fonts_dir):
        # Fallback to system font
        return "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    
    # Prefer bold fonts
    for fn in os.listdir(fonts_dir):
        if "bold" in fn.lower() and fn.lower().endswith((".ttf", ".otf")):
            return os.path.join(fonts_dir, fn)
    
    for fn in os.listdir(fonts_dir):
        if fn.lower().endswith((".ttf", ".otf")):
            return os.path.join(fonts_dir, fn)
    
    return "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def make_overlay_png(path_png, frame_w, frame_h, blocks, font_path):
    """Create a single overlay PNG with all text blocks at CONSISTENT position."""
    img = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Combine all text blocks into one overlay at consistent bottom position
    all_texts = []
    for block in blocks:
        text = block.get("text_da", block.get("text", ""))
        if text:
            all_texts.append(text.replace('\n', ' ').strip())
    
    if not all_texts:
        img.save(path_png)
        return
    
    # Join all text blocks with newlines
    combined_text = '\n'.join(all_texts)
    
    # Use consistent styling - smaller font to fit more text
    font_size = max(36, int(frame_h * 0.045))  # ~4.5% of frame height
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    
    fill = (255, 255, 255)  # White text
    stroke_fill = (0, 0, 0)  # Black stroke
    stroke_width = max(3, font_size // 12)
    
    # Word wrap to fit frame width (with margins)
    margin_x = int(frame_w * 0.05)
    max_width = frame_w - (margin_x * 2)
    
    wrapped_lines = []
    for paragraph in combined_text.split('\n'):
        words = paragraph.split()
        if not words:
            continue
        current_line = words[0]
        for word in words[1:]:
            test = f"{current_line} {word}"
            test_bbox = draw.textbbox((0, 0), test, font=font)
            if test_bbox[2] - test_bbox[0] <= max_width:
                current_line = test
            else:
                wrapped_lines.append(current_line)
                current_line = word
        wrapped_lines.append(current_line)
    
    wrapped = '\n'.join(wrapped_lines)
    
    # Calculate text dimensions
    tb = draw.multiline_textbbox((0, 0), wrapped, font=font, stroke_width=stroke_width, spacing=8, align="center")
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    
    # CONSISTENT POSITION: Center horizontally, fixed distance from bottom
    tx = (frame_w - tw) // 2
    ty = int(frame_h * 0.72) - (th // 2)  # ~72% down, centered vertically around that point
    
    # Ensure not too close to bottom edge
    if ty + th > frame_h - 20:
        ty = frame_h - th - 20
    
    # Draw shadow
    shadow_offset = max(3, font_size // 15)
    draw.multiline_text((tx + shadow_offset, ty + shadow_offset), wrapped, font=font, 
                       fill=(0, 0, 0, 200), spacing=8, align="center")
    
    # Draw main text with stroke
    draw.multiline_text((tx, ty), wrapped, font=font, fill=fill,
                       stroke_width=stroke_width, stroke_fill=stroke_fill,
                       spacing=8, align="center")
    
    img.save(path_png)


def _render_overlay_video(segments, base_video_path, out_path, fonts_dir="fonts"):
    """
    Shared rendering logic: overlay translated segments onto a base video.
    Used by both the two-video and single-video workflows.
    """
    font_path = pick_best_font(fonts_dir)
    print(f"Using font: {font_path}")
    
    tmpdir = tempfile.mkdtemp(prefix="sub_overlays_")
    try:
        base = VideoFileClip(base_video_path)
        frame_w, frame_h = base.w, base.h
        
        clips = [base]
        for i, seg in enumerate(segments):
            start = max(0.0, seg["start"])
            end = min(base.duration, seg["end"])
            if end <= start or not seg.get("blocks"):
                continue
            
            png = os.path.join(tmpdir, f"seg_{i:04d}.png")
            make_overlay_png(png, frame_w, frame_h, seg["blocks"], font_path)
            
            overlay = (ImageClip(png)
                       .set_start(start)
                       .set_end(end)
                       .set_position((0, 0)))
            clips.append(overlay)
            
            texts = [b.get("text_da", b.get("text", ""))[:30] for b in seg["blocks"]]
            print(f"  Segment {i+1}: [{start:.1f}s - {end:.1f}s] {texts}")
        
        final = CompositeVideoClip(clips)
        final.write_videofile(out_path, codec="libx264", audio_codec="aac", fps=base.fps)
        print(f"\n=== Done! Output: {out_path} ===")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def burn_in_subtitles(withtext_path, withouttext_path, out_path="output_da.mp4",
                      fonts_dir="fonts", sample_interval=1.0):
    """
    TWO-VIDEO MODE (legacy):
    1) Use vision model to detect all text and timing from withtext.mp4
    2) Translate to Danish
    3) Overlay onto withouttext.mp4
    """
    # 1) Detect text segments
    print("\n=== Detecting text with Vision AI ===")
    segments = detect_text_segments_from_video(withtext_path, sample_interval=sample_interval)
    
    if not segments:
        print("ERROR: No text detected!")
        return
    
    print(f"\nFound {len(segments)} distinct text segment(s)")
    
    # 2) Translate
    print("\n=== Translating to Danish ===")
    segments = translate_segments_to_danish(segments)
    
    # 3) Render
    print("\n=== Rendering output video ===")
    _render_overlay_video(segments, withouttext_path, out_path, fonts_dir)


def burn_in_subtitles_single(input_video, out_path="output_da.mp4",
                              fonts_dir="fonts", sample_interval=1.0):
    """
    SINGLE-VIDEO MODE (new):
    1) Detect text + timing from the input video BEFORE erasing (captures what/when/where)
    2) Translate detected text to Danish
    3) Erase original subtitles using EraseSubtitles (inpainting)
    4) Overlay translated Danish subtitles onto the clean video
    """
    from erase_subtitles import erase_subtitles

    # 1) FIRST: Extract text while subtitles are still visible
    print("\n" + "=" * 60)
    print("STEP 1/4: Detecting original subtitles with Vision AI")
    print("=" * 60)
    segments = detect_text_segments_from_video(input_video, sample_interval=sample_interval)

    if not segments:
        print("ERROR: No text detected in the input video!")
        return

    print(f"\nFound {len(segments)} distinct text segment(s):")
    for i, seg in enumerate(segments):
        for b in seg.get("blocks", []):
            print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] "
                  f"pos={b.get('position', '?')}: \"{b.get('text', '')}\"")

    # 2) Translate to Danish
    print("\n" + "=" * 60)
    print("STEP 2/4: Translating to Danish")
    print("=" * 60)
    segments = translate_segments_to_danish(segments)

    # 3) Erase original subtitles to produce a clean video
    print("\n" + "=" * 60)
    print("STEP 3/4: Erasing original subtitles (inpainting)")
    print("=" * 60)
    stem = os.path.splitext(os.path.basename(input_video))[0]
    clean_video = os.path.join(os.path.dirname(input_video) or ".", f"{stem}_clean.mp4")
    erase_subtitles(input_video, clean_video)

    # 4) Overlay translated text onto the clean video
    print("\n" + "=" * 60)
    print("STEP 4/4: Burning in Danish subtitles onto clean video")
    print("=" * 60)
    _render_overlay_video(segments, clean_video, out_path, fonts_dir)

    # Optional: remove intermediate clean video
    # os.remove(clean_video)
    print(f"\nAll done! Final output: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Burn translated Danish subtitles into a video."
    )
    parser.add_argument(
        "--single", type=str, default=None, metavar="VIDEO",
        help="Single-video mode: provide one video with subtitles. "
             "Subtitles will be detected, translated, erased, and "
             "replaced with Danish subtitles."
    )
    parser.add_argument(
        "--withtext", type=str, default="withtext2.mp4",
        help="(Two-video mode) Video with original subtitles for text detection."
    )
    parser.add_argument(
        "--withouttext", type=str, default="withouttext2.mp4",
        help="(Two-video mode) Clean video without subtitles."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output_da.mp4",
        help="Output video path."
    )
    parser.add_argument(
        "--fonts", type=str, default="fonts",
        help="Directory containing font files."
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Frame sampling interval in seconds for text detection."
    )
    args = parser.parse_args()

    if args.single:
        # New single-video mode
        burn_in_subtitles_single(
            input_video=args.single,
            out_path=args.output,
            fonts_dir=args.fonts,
            sample_interval=args.interval,
        )
    else:
        # Legacy two-video mode
        burn_in_subtitles(
            withtext_path=args.withtext,
            withouttext_path=args.withouttext,
            out_path=args.output,
            fonts_dir=args.fonts,
            sample_interval=args.interval,
        )
