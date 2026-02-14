"""
Cog Predictor for Subtitle Translation Pipeline.

Takes a video with hardcoded English subtitles, detects text via Vision AI,
translates to target language, erases original subtitles via E2FGVI inpainting,
and overlays translated text.
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path

from cog import BasePredictor, Input, Path as CogPath, Secret

# Ensure EraseSubtitles is on path
_ROOT = os.path.dirname(os.path.abspath(__file__))
_ERASE_DIR = os.path.join(_ROOT, "EraseSubtitles")


class Predictor(BasePredictor):
    def setup(self):
        """Download model weights (if needed) and load E2FGVI onto GPU."""
        import torch
        import subprocess

        # --- Download model weights if not present ---
        weights = [
            {
                "path": os.path.join(_ERASE_DIR, "E2FGVI", "release_model", "E2FGVI-CVPR22.pth"),
                "gdrive_id": "10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3",
            },
            {
                "path": os.path.join(_ERASE_DIR, "E2FGVI", "release_model", "E2FGVI-HQ-CVPR22.pth"),
                "gdrive_id": "1jGj_2IXLK_gf_MZtfHPnob5cqIOPgSaO",
            },
            {
                "path": os.path.join(_ERASE_DIR, "text_detection", "weights", "new_ctpn_ep09_0.0420_0.0198_0.0618.pth"),
                "gdrive_id": "1LX1v1aAPYTUgOFYnNTvBUOsEIF-FVP-I",
            },
        ]
        # CRAFT weight is downloaded by the CRAFT module itself via torch.hub

        for w in weights:
            if not os.path.isfile(w["path"]):
                os.makedirs(os.path.dirname(w["path"]), exist_ok=True)
                print(f"[setup] Downloading {os.path.basename(w['path'])}...")
                subprocess.run(
                    ["python3", "-m", "gdown", w["gdrive_id"], "-O", w["path"]],
                    check=True,
                )

        # --- Load inpainting model onto GPU ---
        sys.path.insert(0, _ERASE_DIR)
        os.chdir(_ERASE_DIR)

        from inpaint import set_up_model
        self.model, self.device = set_up_model()
        print(f"[setup] E2FGVI loaded on {self.device}")

        os.chdir(_ROOT)

    def predict(
        self,
        video: CogPath = Input(description="Input video with hardcoded subtitles"),
        target_language: str = Input(
            description="Target language code for translation",
            default="da",
            choices=["da", "de", "fr", "es", "it", "nl", "pt", "sv", "no"],
        ),
        openrouter_api_key: Secret = Input(
            description="OpenRouter API key for Vision AI text detection and LLM translation"
        ),
        sample_interval: float = Input(
            description="Frame sampling interval in seconds for text detection (lower = more accurate, slower)",
            default=1.0,
            ge=0.5,
            le=5.0,
        ),
    ) -> CogPath:
        """Run the full subtitle translation pipeline."""
        import json
        import re
        import base64
        import numpy as np
        import cv2
        import requests
        from PIL import Image, ImageDraw, ImageFont
        from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip

        api_key = openrouter_api_key.get_secret_value()

        # --- Copy input video to working directory ---
        work_dir = tempfile.mkdtemp(prefix="subtitle_")
        input_path = os.path.join(work_dir, "input.mp4")
        shutil.copy2(str(video), input_path)

        # ============================================================
        # STEP 1: Detect text segments from video
        # ============================================================
        print("\n" + "=" * 60)
        print("STEP 1/4: Detecting original subtitles with Vision AI")
        print("=" * 60)
        segments = self._detect_text_segments(input_path, api_key, sample_interval)

        if not segments:
            print("No text detected — returning original video.")
            return CogPath(str(video))

        print(f"\nFound {len(segments)} distinct text segment(s)")

        # ============================================================
        # STEP 2: Translate to target language
        # ============================================================
        print("\n" + "=" * 60)
        print(f"STEP 2/4: Translating to {target_language}")
        print("=" * 60)
        segments = self._translate_segments(segments, target_language, api_key)

        # ============================================================
        # STEP 3: Erase original subtitles (GPU-accelerated inpainting)
        # ============================================================
        print("\n" + "=" * 60)
        print("STEP 3/4: Erasing original subtitles (inpainting)")
        print("=" * 60)
        clean_path = os.path.join(work_dir, "clean.mp4")
        self._erase_subtitles(input_path, clean_path)

        # ============================================================
        # STEP 4: Overlay translated subtitles
        # ============================================================
        print("\n" + "=" * 60)
        print("STEP 4/4: Burning in translated subtitles")
        print("=" * 60)
        output_path = os.path.join(work_dir, "output.mp4")
        self._render_overlay(segments, clean_path, output_path)

        print(f"\nDone! Output: {output_path}")
        return CogPath(output_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _frame_to_base64(self, frame_bgr):
        _, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode("utf-8")

    def _extract_text_with_vision(self, frame_bgr, api_key):
        import requests, json, re
        base64_img = self._frame_to_base64(frame_bgr)

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
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "google/gemini-2.0-flash-001",
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]}],
                    "max_tokens": 1000
                },
                timeout=30,
            )
            if response.status_code != 200:
                return []
            content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            json_match = re.search(r'\[[\s\S]*?\]', content)
            if json_match:
                return json.loads(json_match.group())
            return []
        except Exception as e:
            print(f"  Vision API exception: {e}")
            return []

    def _normalize(self, text):
        import re
        return re.sub(r'\s+', ' ', text.lower().strip())

    def _texts_are_similar(self, t1, t2):
        n1, n2 = self._normalize(t1), self._normalize(t2)
        if not n1 or not n2:
            return False
        if n1 in n2 or n2 in n1:
            return True
        words1, words2 = set(n1.split()), set(n2.split())
        if not words1 or not words2:
            return False
        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap > 0.5

    def _detect_text_segments(self, video_path, api_key, interval):
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        segments = []
        current_segment = None
        t = 0.0

        while t <= duration:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                t += interval
                continue

            blocks = self._extract_text_with_vision(frame, api_key)
            combined = " ".join(b.get("text", "") for b in blocks).strip()

            print(f"  Analyzing t={t:.1f}s... found {len(blocks)} text block(s)")
            for b in blocks:
                print(f"    - {b.get('position', '?')}: \"{b.get('text', '')[:50]}...\"" if len(b.get('text', '')) > 50 else f"    - {b.get('position', '?')}: \"{b.get('text', '')}\"")

            if blocks:
                if current_segment and self._texts_are_similar(
                    " ".join(b.get("text", "") for b in current_segment["blocks"]),
                    combined
                ):
                    current_segment["end"] = t + interval
                else:
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = {"start": t, "end": t + interval, "blocks": blocks}
            else:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = None

            t += interval

        if current_segment:
            segments.append(current_segment)

        return segments

    def _translate_segments(self, segments, lang, api_key):
        import requests
        lang_names = {
            "da": "Danish", "de": "German", "fr": "French", "es": "Spanish",
            "it": "Italian", "nl": "Dutch", "pt": "Portuguese", "sv": "Swedish", "no": "Norwegian"
        }
        lang_name = lang_names.get(lang, lang)

        for seg in segments:
            for block in seg.get("blocks", []):
                text = block.get("text", "")
                if not text:
                    continue
                prompt = f"""Translate the following English text to natural, idiomatic {lang_name}.
IMPORTANT:
- Do NOT translate word-for-word literally
- The translation should sound like a native {lang_name} speaker wrote it
- Keep the same meaning and tone but use natural phrasing
- Keep it concise for video subtitles

English text:
"{text}"

Respond with ONLY the {lang_name} translation, nothing else."""

                try:
                    resp = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={"model": "google/gemini-2.0-flash-001", "messages": [{"role": "user", "content": prompt}], "max_tokens": 500},
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        translated = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                        block["text_da"] = translated or text
                    else:
                        block["text_da"] = text
                except Exception:
                    block["text_da"] = text

                short = text[:40] + "..." if len(text) > 40 else text
                short_da = block["text_da"][:40] + "..." if len(block["text_da"]) > 40 else block["text_da"]
                print(f"  '{short}' -> '{short_da}'")

        return segments

    def _erase_subtitles(self, input_video, output_video):
        """Run E2FGVI inpainting with pre-loaded model."""
        original_cwd = os.getcwd()
        original_path = sys.path.copy()

        try:
            os.chdir(_ERASE_DIR)
            if _ERASE_DIR not in sys.path:
                sys.path.insert(0, _ERASE_DIR)

            from preprocessing import gen_image_frames, seg_imgs
            from detectText import get_coords
            from splitRegion import gen_regions
            from inpaint import gen_frames_and_masks, inpaint_main, merge

            ip_path = os.path.join(_ERASE_DIR, "Input", "Video") + os.sep
            a_path = os.path.join(_ERASE_DIR, "Input", "Audio") + os.sep
            os.makedirs(ip_path, exist_ok=True)
            os.makedirs(a_path, exist_ok=True)

            video_filename = os.path.basename(input_video)
            staged = os.path.join(ip_path, video_filename)
            shutil.copy2(input_video, staged)

            inpainted_dir = os.path.join(_ERASE_DIR, "Output", "Inpainted")
            os.makedirs(inpainted_dir, exist_ok=True)

            print("  Extracting frames...")
            images = gen_image_frames(video_filename, ip_path, a_path)
            masks = seg_imgs(images)
            num_of_frames = len(masks)
            print(f"  Frames: {num_of_frames}")

            print("  Detecting subtitle region...")
            coords = get_coords(num_of_frames, masks)
            print(f"  Region: {coords}")

            if not coords:
                shutil.copy2(input_video, output_video)
                return

            h, w = 240, 432
            new_coords, num_of_splits, final_images, final_masks = gen_regions(h, w, images, masks, coords)

            print(f"  Inpainting {num_of_splits} splits...")
            iframes, imasks = gen_frames_and_masks(final_images, final_masks)
            comp_frames = inpaint_main(iframes, imasks)
            inpainted_frames = merge(num_of_frames, num_of_splits, new_coords, comp_frames, images)

            # Write video
            im = inpainted_frames[0]
            h_out, w_out = im.shape[:2]
            temp_video = os.path.join(inpainted_dir, video_filename)
            writer = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*"MP4V"), 30, (w_out, h_out))
            for f in inpainted_frames:
                writer.write(f)
            writer.release()

            # Re-attach audio
            from moviepy.editor import VideoFileClip, AudioFileClip
            video_stem = video_filename[:-4]
            vc = VideoFileClip(temp_video)
            audio_path = os.path.join(a_path, video_stem + ".mp3")
            if os.path.isfile(audio_path):
                ac = AudioFileClip(audio_path)
                fc = vc.set_audio(ac)
            else:
                fc = vc
            fc.write_videofile(output_video, codec="libx264", audio_codec="aac")
            vc.close()
            fc.close()
            print("  Inpainting complete!")

        finally:
            os.chdir(original_cwd)
            sys.path = original_path

    def _render_overlay(self, segments, base_video_path, out_path):
        """Overlay translated subtitles onto the clean video."""
        from PIL import Image, ImageDraw, ImageFont
        from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip

        # Use bundled fonts
        fonts_dir = os.path.join(_ROOT, "fonts")
        font_path = self._pick_font(fonts_dir)
        print(f"  Using font: {font_path}")

        tmpdir = tempfile.mkdtemp(prefix="overlays_")
        try:
            base = VideoFileClip(base_video_path)
            fw, fh = base.w, base.h
            clips = [base]

            for i, seg in enumerate(segments):
                start = max(0.0, seg["start"])
                end = min(base.duration, seg["end"])
                if end <= start or not seg.get("blocks"):
                    continue

                png = os.path.join(tmpdir, f"seg_{i:04d}.png")
                self._make_overlay(png, fw, fh, seg["blocks"], font_path)
                overlay = ImageClip(png).set_start(start).set_end(end).set_position((0, 0))
                clips.append(overlay)

            final = CompositeVideoClip(clips)
            final.write_videofile(out_path, codec="libx264", audio_codec="aac", fps=base.fps)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _pick_font(self, fonts_dir):
        if os.path.isdir(fonts_dir):
            for fn in os.listdir(fonts_dir):
                if "bold" in fn.lower() and fn.lower().endswith((".ttf", ".otf")):
                    return os.path.join(fonts_dir, fn)
            for fn in os.listdir(fonts_dir):
                if fn.lower().endswith((".ttf", ".otf")):
                    return os.path.join(fonts_dir, fn)
        # Linux fallback
        for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                   "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
            if os.path.isfile(p):
                return p
        return None

    def _make_overlay(self, path_png, fw, fh, blocks, font_path):
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        all_texts = []
        for b in blocks:
            text = b.get("text_da", b.get("text", ""))
            if text:
                all_texts.append(text.replace('\n', ' ').strip())

        if not all_texts:
            img.save(path_png)
            return

        combined = '\n'.join(all_texts)
        font_size = max(36, int(fh * 0.045))
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

        stroke_width = max(3, font_size // 12)
        margin_x = int(fw * 0.05)
        max_w = fw - margin_x * 2

        wrapped_lines = []
        for para in combined.split('\n'):
            words = para.split()
            if not words:
                continue
            line = words[0]
            for w in words[1:]:
                test = f"{line} {w}"
                tb = draw.textbbox((0, 0), test, font=font)
                if tb[2] - tb[0] <= max_w:
                    line = test
                else:
                    wrapped_lines.append(line)
                    line = w
            wrapped_lines.append(line)

        wrapped = '\n'.join(wrapped_lines)
        tb = draw.multiline_textbbox((0, 0), wrapped, font=font, stroke_width=stroke_width, spacing=8, align="center")
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        tx = (fw - tw) // 2
        ty = int(fh * 0.72) - th // 2
        if ty + th > fh - 20:
            ty = fh - th - 20

        shadow = max(3, font_size // 15)
        draw.multiline_text((tx + shadow, ty + shadow), wrapped, font=font, fill=(0, 0, 0, 200), spacing=8, align="center")
        draw.multiline_text((tx, ty), wrapped, font=font, fill=(255, 255, 255), stroke_width=stroke_width, stroke_fill=(0, 0, 0), spacing=8, align="center")
        img.save(path_png)
