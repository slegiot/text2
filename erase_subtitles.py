"""
Wrapper module for the EraseSubtitles pipeline.
Provides a clean interface to erase hardcoded subtitles from a video file
using CRAFT text detection + E2FGVI video inpainting.

Requirements:
  - EraseSubtitles repo cloned into ./EraseSubtitles/
  - Text detection weights in EraseSubtitles/text_detection/weights/
  - E2FGVI model in EraseSubtitles/E2FGVI/release_model/
  - PyTorch installed
"""

import os
import sys
import shutil
import cv2
import numpy as np

# Root of the EraseSubtitles repo (relative to this file)
_ERASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EraseSubtitles")


def _check_prerequisites():
    """Validate that the EraseSubtitles repo and checkpoints exist."""
    if not os.path.isdir(_ERASE_DIR):
        raise FileNotFoundError(
            f"EraseSubtitles repo not found at {_ERASE_DIR}. "
            "Run: git clone https://github.com/Rats20/EraseSubtitles.git"
        )

    weights_dir = os.path.join(_ERASE_DIR, "text_detection", "weights")
    if not os.path.isdir(weights_dir) or not os.listdir(weights_dir):
        raise FileNotFoundError(
            f"Text detection weights not found at {weights_dir}. "
            "Download from: https://drive.google.com/drive/folders/1ZeimKwzWYDWxHOV6-ES78T5W_kJJAEvP"
        )

    model_dir = os.path.join(_ERASE_DIR, "E2FGVI", "release_model")
    if not os.path.isdir(model_dir) or not os.listdir(model_dir):
        raise FileNotFoundError(
            f"E2FGVI model not found at {model_dir}. "
            "Download from: https://drive.google.com/drive/folders/1duoBn3eHIDpW4hnMmpkZYN4cKtgpLwbU"
        )


def erase_subtitles(input_video: str, output_video: str, fps: int = 30) -> str:
    """
    Remove hardcoded subtitles from a video using the EraseSubtitles pipeline.

    Pipeline:
      1. Extract frames + audio from input video
      2. Color-segment frames to create subtitle masks
      3. CRAFT text detection to find subtitle region bounding box
      4. Split subtitle region into 240x432 tiles for inpainting
      5. E2FGVI video inpainting to remove subtitles
      6. Reassemble frames + audio into output video

    Args:
        input_video: Path to the input video with subtitles.
        output_video: Path to save the clean (subtitle-free) output.
        fps: Frame rate for the output video.

    Returns:
        Absolute path to the output video.
    """
    _check_prerequisites()

    input_video = os.path.abspath(input_video)
    output_video = os.path.abspath(output_video)

    if not os.path.isfile(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")

    # --- Change working directory to EraseSubtitles so relative imports work ---
    original_cwd = os.getcwd()
    original_sys_path = sys.path.copy()

    try:
        os.chdir(_ERASE_DIR)
        if _ERASE_DIR not in sys.path:
            sys.path.insert(0, _ERASE_DIR)

        # Import the EraseSubtitles modules (must happen after chdir + sys.path)
        from preprocessing import gen_image_frames, seg_imgs
        from detectText import get_coords
        from splitRegion import gen_regions
        from inpaint import gen_frames_and_masks, inpaint_main, merge

        # --- Set up Input directories ---
        ip_path = os.path.join(_ERASE_DIR, "Input", "Video") + os.sep
        a_path = os.path.join(_ERASE_DIR, "Input", "Audio") + os.sep
        os.makedirs(ip_path, exist_ok=True)
        os.makedirs(a_path, exist_ok=True)

        # Copy input video into the expected Input/Video/ location
        video_filename = os.path.basename(input_video)
        staged_video = os.path.join(ip_path, video_filename)
        if os.path.abspath(input_video) != os.path.abspath(staged_video):
            shutil.copy2(input_video, staged_video)

        # --- Set up Output directories ---
        inpainted_dir = os.path.join(_ERASE_DIR, "Output", "Inpainted")
        output_dir = os.path.join(_ERASE_DIR, "Output")
        os.makedirs(inpainted_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # ============================================================
        # 1. Preprocessing: extract frames + audio
        # ============================================================
        print("\n=== [EraseSubtitles] Preprocessing ===")
        images = gen_image_frames(video_filename, ip_path, a_path)
        masks = seg_imgs(images)
        num_of_frames = len(masks)
        print(f"  Frames extracted: {num_of_frames}")

        # ============================================================
        # 2. Text detection: find subtitle bounding box
        # ============================================================
        print("\n=== [EraseSubtitles] Detecting subtitle region ===")
        max_height, max_width = images[0].shape[:2]
        coords = get_coords(num_of_frames, masks)
        print(f"  Subtitle region coords: {coords}")

        if coords == []:
            print("  No subtitles detected — copying input as-is.")
            shutil.copy2(input_video, output_video)
            return output_video

        # ============================================================
        # 3. Split subtitle region into tiles
        # ============================================================
        h, w = 240, 432
        print(f"\n=== [EraseSubtitles] Splitting to {h}x{w} tiles ===")
        new_coords, num_of_splits, final_images, final_masks = gen_regions(
            h, w, images, masks, coords
        )
        print(f"  Split coords: {new_coords}")
        print(f"  Number of splits: {num_of_splits}")

        # ============================================================
        # 4. Inpainting: remove subtitles
        # ============================================================
        print("\n=== [EraseSubtitles] Inpainting (this may take a while) ===")
        iframes, imasks = gen_frames_and_masks(final_images, final_masks)
        comp_frames = inpaint_main(iframes, imasks)
        inpainted_frames = merge(num_of_frames, num_of_splits, new_coords, comp_frames, images)
        print("  Inpainting complete!")

        # ============================================================
        # 5. Write inpainted video (no audio)
        # ============================================================
        im = inpainted_frames[0]
        h_out, w_out = im.shape[:2]
        size = (w_out, h_out)

        temp_video_path = os.path.join(inpainted_dir, video_filename)
        out_writer = cv2.VideoWriter(
            temp_video_path, cv2.VideoWriter_fourcc(*"MP4V"), fps, size
        )
        print(f"\n=== [EraseSubtitles] Writing clean video ({num_of_frames} frames) ===")
        for i in range(num_of_frames):
            out_writer.write(inpainted_frames[i])
        out_writer.release()

        # ============================================================
        # 6. Re-attach audio
        # ============================================================
        print("=== [EraseSubtitles] Re-attaching audio ===")
        from moviepy.editor import VideoFileClip, AudioFileClip

        video_stem = video_filename[:-4]  # remove .mp4
        video_clip = VideoFileClip(temp_video_path)
        audio_path = os.path.join(a_path, video_stem + ".mp3")

        if os.path.isfile(audio_path):
            audio_clip = AudioFileClip(audio_path)
            final_clip = video_clip.set_audio(audio_clip)
        else:
            print("  Warning: no audio file found, output will have no audio.")
            final_clip = video_clip

        final_clip.write_videofile(output_video, codec="libx264", audio_codec="aac")

        # Clean up moviepy resources
        video_clip.close()
        final_clip.close()

        print(f"\n=== [EraseSubtitles] Done! Clean video saved: {output_video} ===")
        return output_video

    finally:
        # Restore original working directory and sys.path
        os.chdir(original_cwd)
        sys.path = original_sys_path


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Remove hardcoded subtitles from a video using EraseSubtitles."
    )
    parser.add_argument("input", help="Path to the input video with subtitles")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output path (default: <input>_clean.mp4)"
    )
    parser.add_argument("--fps", type=int, default=30, help="Output FPS (default: 30)")
    args = parser.parse_args()

    out = args.output or args.input.rsplit(".", 1)[0] + "_clean.mp4"
    erase_subtitles(args.input, out, fps=args.fps)
