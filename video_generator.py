#!/usr/bin/env python3
import argparse
import os
import pandas as pd
from pathlib import Path
from moviepy import *
from moviepy import TextClip, AudioFileClip, ColorClip, CompositeVideoClip, ImageSequenceClip, VideoFileClip, VideoClip, ImageClip
import numpy as np
import platform
import tempfile
from shaderbg import generate_shader_frames, cleanup_frames
from filterbg import process_frames, apply_interlace_filter, apply_scanlines_filter
from PIL import Image


def read_lyrics_csv(csv_path):
    """
    Read lyrics with timestamps from CSV
    """
    df = pd.read_csv(csv_path)
    lyrics_data = []
    
    for _, row in df.iterrows():
        if pd.notna(row.get('start')) and pd.notna(row.get('end')) and row['start'] != '':
            lyrics_data.append({
                'text': row['lyric'],
                'start': float(row['start']),
                'end': float(row['end'])
            })
    
    return lyrics_data


def get_japanese_font():
    """
    Get appropriate Japanese font based on platform
    """
    system = platform.system()
    if system == 'Darwin':  # macOS
        font_names = [
            'ヒラギノ明朝 ProN',
            'Hiragino Sans',
            'Hiragino Kaku Gothic ProN',
            'Arial Unicode MS',
            'Arial'
        ]
        for font_name in font_names:
            try:
                test_clip = TextClip(text='テスト', font=font_name, font_size=12, method='label')
                test_clip.close()
                return font_name
            except:
                continue
    elif system == 'Windows':
        fonts = ['MS-Gothic', 'Yu-Gothic', 'Meiryo', 'Arial']
        for font in fonts:
            try:
                test_clip = TextClip(text='テスト', font=font, font_size=12, method='label')
                test_clip.close()
                return font
            except:
                continue
    else:  # Linux
        fonts = ['Noto-Sans-CJK-JP', 'TakaoGothic', 'IPAGothic', 'DejaVu-Sans']
        for font in fonts:
            try:
                test_clip = TextClip(text='テスト', font=font, font_size=12, method='label')
                test_clip.close()
                return font
            except:
                continue
    
    return 'Arial'  # Default fallback


def calculate_optimal_font_size(text, video_width, font, max_font_size=50, min_font_size=20):
    """
    Calculate optimal font size to fit text in one line without wrapping
    """
    target_width = int(video_width * 0.9)  # 90% of video width
    
    for font_size in range(max_font_size, min_font_size - 1, -2):
        try:
            # Create test clip to measure text width
            test_clip = TextClip(
                text=text,
                font_size=font_size,
                font=font,
                method='label'  # Use label for single line measurement
            )
            
            # Check if text fits within target width
            if test_clip.w <= target_width:
                test_clip.close()
                return font_size
            
            test_clip.close()
        except:
            continue
    
    return min_font_size  # Return minimum size if nothing fits


def create_text_clip(text, duration, video_width, video_height, position_index=0, opacity=1.0, font_size=50, lyric_position='bottom', total_lines=3):
    """
    Create a text clip with specified position and opacity
    position_index: 0 = current (newest), 1 = previous, 2 = oldest
    lyric_position: 'bottom' or 'center' - vertical alignment of lyric block
    total_lines: total number of lines to display (needed for center positioning)
    """
    # Get appropriate font for Japanese text
    font = get_japanese_font()
    
    txt_clip = TextClip(
        text=text, 
        font_size=font_size,
        color='white',
        font=font,
        stroke_color='black',
        stroke_width=2,
        method='label',  # Use label to ensure single line
        text_align='center'
    )
    
    # Apply opacity
    if opacity < 1.0:
        txt_clip = txt_clip.with_opacity(opacity)
    
    # Calculate position based on index and alignment
    vertical_spacing = font_size + 25  # Use font size for spacing calculation
    
    if lyric_position == 'bottom':
        # Stack from bottom up
        bottom_margin = 50
        y_position = video_height - bottom_margin - (position_index + 1) * vertical_spacing
    else:  # center
        # Calculate center position for the entire lyric block, but stack upward like bottom mode
        total_height = total_lines * vertical_spacing - 25  # Remove extra spacing from last line
        block_bottom = video_height // 2 + total_height // 2
        y_position = block_bottom - (position_index + 1) * vertical_spacing
    
    txt_clip = txt_clip.with_position(('center', y_position))
    txt_clip = txt_clip.with_duration(duration)
    
    return txt_clip


def create_title_clip(title, artist, audio_duration, video_width, video_height):
    """
    Create a title clip for top-left corner display
    """
    font = get_japanese_font()
    
    # Create title text
    if artist:
        title_text = f"{title}\n{artist}"
    else:
        title_text = title
    
    title_clip = TextClip(
        text=title_text,
        font_size=24,
        color='white',
        font=font,
        stroke_color='black',
        stroke_width=1,
        method='label',
        text_align='left'
    )
    
    # Position at top-left with margin
    title_clip = title_clip.with_position((20, 20))
    title_clip = title_clip.with_duration(audio_duration)
    
    return title_clip


def create_watermark_clip(watermark_path, video_width, video_height, duration, opacity=0.5, mode='fit'):
    """
    Create a watermark clip with specified opacity and scaling mode
    mode: 'fit' - entire image visible (may have borders)
          'fill' - cover entire frame (may crop image)
    """
    if not os.path.exists(watermark_path):
        print(f"Warning: Watermark file '{watermark_path}' not found")
        return None
    
    try:
        # Load watermark image
        watermark_img = Image.open(watermark_path)
        
        # Convert to RGBA if not already
        if watermark_img.mode != 'RGBA':
            watermark_img = watermark_img.convert('RGBA')
        
        # Calculate aspect ratios
        video_aspect = video_width / video_height
        img_aspect = watermark_img.width / watermark_img.height
        
        if mode == 'fit':
            # Fit mode: entire image visible
            if img_aspect > video_aspect:
                # Image is wider - fit to width
                new_width = video_width
                new_height = int(video_width / img_aspect)
            else:
                # Image is taller - fit to height
                new_height = video_height
                new_width = int(video_height * img_aspect)
            
            # Resize watermark to fit
            watermark_resized = watermark_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create a transparent canvas of video size
            canvas = Image.new('RGBA', (video_width, video_height), (0, 0, 0, 0))
            
            # Paste watermark centered on canvas
            x_offset = (video_width - new_width) // 2
            y_offset = (video_height - new_height) // 2
            canvas.paste(watermark_resized, (x_offset, y_offset), watermark_resized)
            
        else:  # mode == 'fill'
            # Fill mode: cover entire frame (may crop)
            if img_aspect > video_aspect:
                # Image is wider - fit to height (will crop width)
                new_height = video_height
                new_width = int(video_height * img_aspect)
            else:
                # Image is taller - fit to width (will crop height)
                new_width = video_width
                new_height = int(video_width / img_aspect)
            
            # Resize watermark to fill
            watermark_resized = watermark_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Crop to video size from center
            x_offset = (new_width - video_width) // 2
            y_offset = (new_height - video_height) // 2
            
            # Crop the resized image to video dimensions
            canvas = watermark_resized.crop((x_offset, y_offset, x_offset + video_width, y_offset + video_height))
        
        # Convert to numpy array
        watermark_array = np.array(canvas)
        
        # Create ImageClip from array
        watermark_clip = ImageClip(watermark_array, duration=duration)
        
        # Apply opacity
        watermark_clip = watermark_clip.with_opacity(opacity)
        
        print(f"Watermark loaded: {watermark_path}")
        print(f"Original size: {watermark_img.width}x{watermark_img.height}")
        if mode == 'fit':
            print(f"Resized to: {new_width}x{new_height} (fit mode)")
        else:
            print(f"Resized to: {new_width}x{new_height}, cropped to: {video_width}x{video_height} (fill mode)")
        print(f"Opacity: {opacity}")
        
        return watermark_clip
        
    except Exception as e:
        print(f"Error loading watermark: {e}")
        return None


def create_interlaced_video(video_clip, use_scanlines=False):
    """
    Create a video clip with interlace filter applied to each frame
    """
    # Debug video info
    print(f"create_interlaced_video: video_clip.size = {video_clip.size}")
    print(f"create_interlaced_video: video_clip.duration = {video_clip.duration}")
    
    # Get a test frame to check dimensions
    test_frame = video_clip.get_frame(0)
    print(f"Test frame shape: {test_frame.shape}, dtype: {test_frame.dtype}")
    print(f"Test frame min/max: {test_frame.min()}/{test_frame.max()}")
    
    # Track if we've shown debug info
    debug_shown = [False]
    
    # Create a simple interlace effect function
    def apply_simple_interlace(frame):
        """Apply a simple interlace effect"""
        # Debug first frame
        if not debug_shown[0]:
            print(f"apply_simple_interlace called: frame shape={frame.shape}, dtype={frame.dtype}")
            print(f"Frame value range: min={frame.min()}, max={frame.max()}")
            print(f"use_scanlines = {use_scanlines}")
        
        # Work with a copy to avoid modifying the original
        result = frame.copy()
        
        # Apply interlace effect - process every other line
        if use_scanlines:
            # Scanlines: darken odd lines by multiplying
            for y in range(1, result.shape[0], 2):  # Odd lines: 1, 3, 5, ...
                result[y] = result[y] * 0.3  # 70% darker (very visible)
            if not debug_shown[0]:
                print(f"Applied scanlines: odd lines (1, 3, 5...) at 70% darkness")
                print(f"Modified {len(range(1, result.shape[0], 2))} lines out of {result.shape[0]} total lines")
        else:
            # Hard interlace: black lines on odd lines
            for y in range(1, result.shape[0], 2):  # Odd lines: 1, 3, 5, ...
                result[y] = 0  # Black line
            if not debug_shown[0]:
                print(f"Applied hard interlace: odd lines (1, 3, 5...) set to black")
                print(f"Modified {len(range(1, result.shape[0], 2))} lines out of {result.shape[0]} total lines")
        
        # Debug: Check if result is different from input
        if not debug_shown[0]:
            diff = np.abs(result - frame).sum()
            print(f"Total difference between input and output: {diff}")
            print(f"Result value range: min={result.min()}, max={result.max()}")
            debug_shown[0] = True
        
        return result
    
    # Apply the effect using available MoviePy methods
    try:
        # First try fl_image (newer MoviePy)
        if hasattr(video_clip, 'fl_image') and callable(getattr(video_clip, 'fl_image')):
            print("Using fl_image method")
            interlaced_clip = video_clip.fl_image(apply_simple_interlace)
            print("fl_image succeeded")
            return interlaced_clip
        # Then try fl (older MoviePy)
        elif hasattr(video_clip, 'fl') and callable(getattr(video_clip, 'fl')):
            print("Using fl method")
            def make_frame_func(gf, t):
                frame = gf(t)
                return apply_simple_interlace(frame)
            interlaced_clip = video_clip.fl(make_frame_func)
            print("fl succeeded")
            return interlaced_clip
        # Finally, try manual approach
        else:
            print("Using manual VideoClip creation")
            # Store original frame function
            if hasattr(video_clip, 'frame_function'):
                original_frame_function = video_clip.frame_function
            else:
                # Use get_frame as fallback
                original_frame_function = lambda t: video_clip.get_frame(t)
            
            # Create new frame function
            def new_frame_function(t):
                frame = original_frame_function(t)
                return apply_simple_interlace(frame)
            
            # Create a new VideoClip with the modified frame function
            from moviepy import VideoClip
            new_clip = VideoClip(duration=video_clip.duration)
            new_clip.size = video_clip.size
            new_clip.fps = video_clip.fps if hasattr(video_clip, 'fps') else 24
            new_clip.frame_function = new_frame_function
            
            # Copy audio if present
            if hasattr(video_clip, 'audio') and video_clip.audio is not None:
                new_clip.audio = video_clip.audio
            
            return new_clip
            
    except Exception as e:
        print(f"Error applying interlace: {e}")
        import traceback
        traceback.print_exc()
        return video_clip


def create_shader_background_ondemand(shader_path, video_width, video_height, duration, fps=24, apply_interlace=True):
    """
    Create shader background with on-demand frame generation (no temporary files)
    """
    try:
        from wgpu_shadertoy import Shadertoy
    except ImportError:
        print("wgpu-shadertoy not available, falling back to file-based method")
        return create_shader_background_legacy(shader_path, video_width, video_height, duration, fps, apply_interlace)
    
    if not os.path.exists(shader_path):
        print(f"Shader file not found: {shader_path}")
        return None, []
    
    try:
        # Shader begin: Initialize renderer once
        print(f"Shader begin: Initializing renderer {video_width}x{video_height}")
        with open(shader_path, 'r') as f:
            shader_code = f.read()
        
        renderer = Shadertoy(shader_code, resolution=(video_width, video_height), offscreen=True)
        print("Shader renderer initialized successfully")
        
        # Track first frame for debugging
        frame_count = [0]
        
        # Create frame generation function
        def make_shader_frame(t):
            """Render frame: Generate shader frame at time t"""
            try:
                # Generate shader frame for time t
                frame_data = renderer.snapshot(time_float=t)
                
                # Convert to numpy array
                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame_array = frame_array.reshape((video_height, video_width, 4))  # RGBA
                
                # Debug first few frames
                if frame_count[0] < 3:
                    print(f"\nShader frame {frame_count[0]} at t={t:.3f}:")
                    print(f"  Raw array shape: {frame_array.shape}")
                    print(f"  RGBA min/max: {frame_array.min()}/{frame_array.max()}")
                    print(f"  R channel min/max: {frame_array[:,:,0].min()}/{frame_array[:,:,0].max()}")
                    print(f"  G channel min/max: {frame_array[:,:,1].min()}/{frame_array[:,:,1].max()}")
                    print(f"  B channel min/max: {frame_array[:,:,2].min()}/{frame_array[:,:,2].max()}")
                    frame_count[0] += 1
                
                # Convert to RGB and normalize to [0,1]
                frame_rgb = frame_array[:, :, :3].astype(np.float64) / 255.0
                
                # Apply interlace filter if requested (same logic as working mode B)
                if apply_interlace:
                    # Convert back to uint8 for interlace processing
                    frame_uint8 = (frame_rgb * 255).astype(np.uint8)
                    
                    # Apply scanlines effect (darken odd lines)
                    for y in range(1, frame_uint8.shape[0], 2):  # Odd lines: 1, 3, 5, ...
                        frame_uint8[y] = frame_uint8[y] * 0.3  # 70% darker (same as working code)
                    
                    # Convert back to float
                    frame_rgb = frame_uint8.astype(np.float64) / 255.0
                
                return frame_rgb
                
            except Exception as e:
                print(f"Error generating shader frame at t={t}: {e}")
                # Return black frame on error
                return np.zeros((video_height, video_width, 3))
        
        # Test the shader by generating a frame
        print("Testing shader frame generation...")
        test_frame = make_shader_frame(0)
        print(f"Test frame result: shape={test_frame.shape}, min={test_frame.min():.3f}, max={test_frame.max():.3f}")
        
        # Create VideoClip with on-demand frame generation
        print("Creating on-demand shader video clip...")
        shader_clip = VideoClip(duration=duration)
        shader_clip.size = (video_width, video_height)
        shader_clip.fps = fps
        shader_clip.make_frame = make_shader_frame  # Try make_frame instead
        # Also try setting frame_function
        shader_clip.frame_function = make_shader_frame
        
        print("Shader background created successfully (on-demand rendering)")
        return shader_clip, []  # No temp files to cleanup
        
    except Exception as e:
        print(f"Error creating on-demand shader background: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to file-based method")
        return create_shader_background_legacy(shader_path, video_width, video_height, duration, fps, apply_interlace)


def create_shader_background_legacy(shader_path, video_width, video_height, duration, fps=24, apply_interlace=True):
    """
    Legacy shader background creation using temporary files (fallback)
    """
    frame_paths, temp_dir = generate_shader_frames(shader_path, video_width, video_height, duration, fps)
    
    if frame_paths:
        # Apply interlace filter to all frames
        if apply_interlace:
            print("Applying interlace filter to background frames...")
            filters = [{'type': 'interlace', 'params': {'line_height': 1, 'spacing': 1}}]
            # Process frames in-place (overwrite originals)
            process_frames(temp_dir, output_dir=None, filters=filters)
            print("Interlace filter applied")
        
        print("Creating video clip from frames...")
        return ImageSequenceClip(frame_paths, fps=fps), frame_paths
    else:
        return None, []


def create_shader_background(shader_path, video_width, video_height, duration, fps=24, apply_interlace=True):
    """
    Create shader background - automatically choose best method
    """
    # Temporarily use legacy method until on-demand is fixed
    print("Using file-based shader background generation")
    return create_shader_background_legacy(shader_path, video_width, video_height, duration, fps, apply_interlace)
    
    # TODO: Fix on-demand method
    # # Try on-demand method first (more efficient)
    # result = create_shader_background_ondemand(shader_path, video_width, video_height, duration, fps, apply_interlace)
    # if result[0] is not None:
    #     return result
    # 
    # # Fallback to legacy method
    # print("Using legacy file-based shader background generation")
    # return create_shader_background_legacy(shader_path, video_width, video_height, duration, fps, apply_interlace)


def create_video_with_lyrics_mode_a(audio_path, lyrics_data, output_path, video_width=1920, video_height=1080, title=None, artist=None, shader_path=None, filter_type='none', lyric_lines=3, lyric_position='bottom', watermark_path=None, watermark_opacity=0.5, watermark_mode='fit', test_mode=False):
    """
    Create video with audio and synchronized lyrics
    """
    print(f"Loading audio: {audio_path}")
    audio = AudioFileClip(str(audio_path))
    
    # Limit duration in test mode
    if test_mode:
        print("TEST MODE: Limiting video to 3 seconds")
        audio = audio.subclipped(0, min(3, audio.duration))
        # Filter lyrics to only include those in the first 3 seconds
        lyrics_data = [lyric for lyric in lyrics_data if lyric['start'] < 3]
    
    # Detect font once
    font = get_japanese_font()
    print(f"Using font: {font}")
    
    # Pre-calculate optimal font sizes for all lyrics to improve performance
    print("Calculating optimal font sizes for all lyrics...")
    font_sizes = {}
    for i, lyric in enumerate(lyrics_data):
        font_sizes[i] = calculate_optimal_font_size(lyric['text'], video_width, font)
        if font_sizes[i] < 50:
            print(f"Adjusted font size for '{lyric['text'][:30]}...': {font_sizes[i]}px")
    
    # Create background (shader or black)
    temp_frame_paths = []
    if shader_path:
        print("Creating shader background...")
        shader_result = create_shader_background(shader_path, video_width, video_height, audio.duration, apply_interlace=(filter_type == 'interlace'))
        if shader_result is not None and shader_result[0] is not None:
            background, temp_frame_paths = shader_result
        else:
            # Fallback to black background
            background = ColorClip(
                size=(video_width, video_height),
                color=(0, 0, 0),
                duration=audio.duration
            )
    else:
        # Default black background
        background = ColorClip(
            size=(video_width, video_height),
            color=(0, 0, 0),
            duration=audio.duration
        )
    
    # Create text clips with continuous history display (variable number of sections)
    text_clips = []
    
    # Calculate opacity levels for each position
    opacity_levels = []
    for pos in range(lyric_lines):
        if pos == 0:
            opacity_levels.append(1.0)  # Current lyric: full opacity
        else:
            # Gradually decrease opacity for older lyrics
            # Scale from 1.0 to 0.25 across all positions
            if lyric_lines > 1:
                opacity = 1.0 - (pos / (lyric_lines - 1)) * 0.75  # 1.0 to 0.25
            else:
                opacity = 1.0
            opacity_levels.append(max(0.25, opacity))  # Ensure minimum 0.25
    
    for i, lyric in enumerate(lyrics_data):
        start_time = lyric['start']
        end_time = lyric['end']
        
        # Create clips for each position this lyric will occupy
        for position in range(lyric_lines):
            # Skip if this lyric won't be shown at this position
            if i + position >= len(lyrics_data):
                continue
            
            # Calculate when this lyric appears at this position
            if position == 0:
                # Current position: shows from its start
                clip_start = start_time
            else:
                # Historical positions: shows from when newer lyrics push it up
                if i + position < len(lyrics_data):
                    clip_start = lyrics_data[i + position]['start']
                else:
                    continue
            
            # Calculate when this lyric disappears from this position
            if i + position + 1 < len(lyrics_data):
                clip_end = lyrics_data[i + position + 1]['start']
            else:
                clip_end = audio.duration
            
            # Skip if duration is invalid
            if clip_end <= clip_start:
                continue
            
            position_name = ['Bottom', 'Middle', 'Top', '4th', '5th'][position] if position < 5 else f'{position+1}th'
            print(f"Lyric {i+1}: '{lyric['text'][:30]}...' - {position_name}: {clip_start:.2f}s -> {clip_end:.2f}s (duration: {clip_end - clip_start:.2f}s)")
            
            clip = create_text_clip(
                lyric['text'],
                clip_end - clip_start,
                video_width,
                video_height,
                position_index=position,
                opacity=opacity_levels[position] if position < len(opacity_levels) else 0.25,
                font_size=font_sizes[i],
                lyric_position=lyric_position,
                total_lines=lyric_lines
            )
            clip = clip.with_start(clip_start)
            text_clips.append(clip)
        
        clips_created = min(lyric_lines, len(lyrics_data) - i)
        print(f"Total clips for lyric {i+1}: {clips_created}")
    
    print(f"\nTotal text clips created: {len(text_clips)}")
    print(f"Audio duration: {audio.duration:.2f}s")
    
    # Create title clip if title is provided
    all_clips = [background] + text_clips
    if title:
        title_clip = create_title_clip(title, artist, audio.duration, video_width, video_height)
        all_clips.append(title_clip)
        print(f"Added title: {title}" + (f" / {artist}" if artist else ""))
    
    # Add watermark if provided
    if watermark_path:
        watermark_clip = create_watermark_clip(watermark_path, video_width, video_height, audio.duration, watermark_opacity, watermark_mode)
        if watermark_clip:
            all_clips.append(watermark_clip)
    
    # Composite all clips
    final_video = CompositeVideoClip(all_clips)
    final_video = final_video.with_audio(audio)
    
    # Write output video
    print(f"Writing video to: {output_path}")
    final_video.write_videofile(
        str(output_path),
        fps=24,
        codec='libx264',
        audio_codec='aac'
    )
    
    # Cleanup
    audio.close()
    final_video.close()
    
    # Clean up temporary shader frames
    if temp_frame_paths:
        print("Cleaning up temporary files...")
        cleanup_frames(temp_frame_paths)


def create_video_with_lyrics_mode_b(video_path, lyrics_data, output_path, title=None, artist=None, filter_type='none', lyric_lines=3, lyric_position='bottom', watermark_path=None, watermark_opacity=0.5, watermark_mode='fit', test_mode=False):
    """
    Create video by compositing lyrics onto existing video
    Mode B: Video + Lyrics -> Composite Video
    """
    print(f"Loading video: {video_path}")
    video = VideoFileClip(str(video_path))
    
    # Limit duration in test mode
    if test_mode:
        print("TEST MODE: Limiting video to 3 seconds")
        video = video.subclipped(0, min(3, video.duration))
        # Filter lyrics to only include those in the first 3 seconds
        lyrics_data = [lyric for lyric in lyrics_data if lyric['start'] < 3]
    
    video_width, video_height = video.size
    print(f"Video dimensions: {video_width}x{video_height}")
    print(f"Video duration: {video.duration:.2f}s")
    
    # Debug: Check original video brightness
    test_frame = video.get_frame(0)
    print(f"Original video frame info: shape={test_frame.shape}, dtype={test_frame.dtype}, min={test_frame.min()}, max={test_frame.max()}")
    
    # Check if video needs brightness correction
    if test_frame.max() < 1.0 and test_frame.dtype in [np.float32, np.float64]:
        # Video is in float format with very low values
        print(f"WARNING: Video appears to be incorrectly scaled. Max value is {test_frame.max()}")
        print("Attempting to correct brightness...")
        
        # Create a brightness correction function
        def correct_brightness(frame):
            # If frame max is less than 1.0 but greater than 0, it might be incorrectly scaled
            if frame.max() < 1.0 and frame.max() > 0:
                # Scale up to proper range
                scale_factor = 255.0 if frame.max() < 0.5 else 1.0
                return np.clip(frame * scale_factor, 0, 255).astype(np.uint8)
            return frame
        
        # Apply brightness correction
        video = video.fl_image(correct_brightness)
        
        # Check again
        test_frame_corrected = video.get_frame(0)
        print(f"Corrected video frame: min={test_frame_corrected.min()}, max={test_frame_corrected.max()}")
    
    # Apply filter to video if requested
    if filter_type == 'interlace':
        print("Applying interlace filter to video...")
        # Use scanlines for dark videos
        use_scanlines = True  # Always use scanlines for now
        if use_scanlines:
            print("Using scanlines filter")
        video = create_interlaced_video(video, use_scanlines=use_scanlines)
        print("Interlace filter applied")
    
    # Detect font once
    font = get_japanese_font()
    print(f"Using font: {font}")
    
    # Pre-calculate optimal font sizes for all lyrics to improve performance
    print("Calculating optimal font sizes for all lyrics...")
    font_sizes = {}
    for i, lyric in enumerate(lyrics_data):
        font_sizes[i] = calculate_optimal_font_size(lyric['text'], video_width, font)
        if font_sizes[i] < 50:
            print(f"Adjusted font size for '{lyric['text'][:30]}...': {font_sizes[i]}px")
    
    # Create text clips with continuous history display (variable number of sections)
    text_clips = []
    
    # Calculate opacity levels for each position
    opacity_levels = []
    for pos in range(lyric_lines):
        if pos == 0:
            opacity_levels.append(1.0)  # Current lyric: full opacity
        else:
            # Gradually decrease opacity for older lyrics
            # Scale from 1.0 to 0.25 across all positions
            if lyric_lines > 1:
                opacity = 1.0 - (pos / (lyric_lines - 1)) * 0.75  # 1.0 to 0.25
            else:
                opacity = 1.0
            opacity_levels.append(max(0.25, opacity))  # Ensure minimum 0.25
    
    for i, lyric in enumerate(lyrics_data):
        start_time = lyric['start']
        end_time = lyric['end']
        
        # Create clips for each position this lyric will occupy
        for position in range(lyric_lines):
            # Skip if this lyric won't be shown at this position
            if i + position >= len(lyrics_data):
                continue
            
            # Calculate when this lyric appears at this position
            if position == 0:
                # Current position: shows from its start
                clip_start = start_time
            else:
                # Historical positions: shows from when newer lyrics push it up
                if i + position < len(lyrics_data):
                    clip_start = lyrics_data[i + position]['start']
                else:
                    continue
            
            # Calculate when this lyric disappears from this position
            if i + position + 1 < len(lyrics_data):
                clip_end = lyrics_data[i + position + 1]['start']
            else:
                clip_end = video.duration
            
            # Skip if duration is invalid
            if clip_end <= clip_start:
                continue
            
            position_name = ['Bottom', 'Middle', 'Top', '4th', '5th'][position] if position < 5 else f'{position+1}th'
            print(f"Lyric {i+1}: '{lyric['text'][:30]}...' - {position_name}: {clip_start:.2f}s -> {clip_end:.2f}s (duration: {clip_end - clip_start:.2f}s)")
            
            clip = create_text_clip(
                lyric['text'],
                clip_end - clip_start,
                video_width,
                video_height,
                position_index=position,
                opacity=opacity_levels[position] if position < len(opacity_levels) else 0.25,
                font_size=font_sizes[i],
                lyric_position=lyric_position,
                total_lines=lyric_lines
            )
            clip = clip.with_start(clip_start)
            text_clips.append(clip)
        
        clips_created = min(lyric_lines, len(lyrics_data) - i)
        print(f"Total clips for lyric {i+1}: {clips_created}")
    
    print(f"\nTotal text clips created: {len(text_clips)}")
    print(f"Video duration: {video.duration:.2f}s")
    
    # Create title clip if title is provided
    all_clips = [video] + text_clips
    if title:
        title_clip = create_title_clip(title, artist, video.duration, video_width, video_height)
        all_clips.append(title_clip)
        print(f"Added title: {title}" + (f" / {artist}" if artist else ""))
    
    # Add watermark if provided
    if watermark_path:
        watermark_clip = create_watermark_clip(watermark_path, video_width, video_height, video.duration, watermark_opacity, watermark_mode)
        if watermark_clip:
            all_clips.append(watermark_clip)
    
    # Composite all clips
    final_video = CompositeVideoClip(all_clips)
    
    # Write output video
    print(f"Writing video to: {output_path}")
    final_video.write_videofile(
        str(output_path),
        fps=24,
        codec='libx264',
        audio_codec='aac'
    )
    
    # Cleanup
    video.close()
    final_video.close()


def main():
    parser = argparse.ArgumentParser(description='Generate video with synchronized lyrics')
    parser.add_argument('input_file', help='Path to input file (MP3 for mode A, video for mode B)')
    parser.add_argument('--mode', choices=['A', 'B'], default='A', 
                       help='Mode A: MP3+Text->Video, Mode B: Video+Lyrics->Composite (default: A)')
    parser.add_argument('--width', type=int, default=1920, help='Video width for mode A (default: 1920)')
    parser.add_argument('--height', type=int, default=1080, help='Video height for mode A (default: 1080)')
    parser.add_argument('--title', help='Song title (default: input filename without extension)')
    parser.add_argument('--artist', help='Artist name')
    parser.add_argument('--shader', help='Path to shader file for background (mode A only)')
    parser.add_argument('--lyrics', help='Path to lyrics CSV file (if not default location)')
    parser.add_argument('--output', help='Output video path (default: auto-generated)')
    parser.add_argument('--filter', choices=['none', 'interlace'], default='none', 
                       help='Video filter to apply: none (default) or interlace')
    parser.add_argument('--lyric-lines', type=int, default=3, help='Number of lyric lines to display (including history) (default: 3)')
    parser.add_argument('--lyric-position', choices=['bottom', 'center'], default='bottom',
                       help='Vertical position of lyrics: bottom (default) or center')
    parser.add_argument('--watermark', help='Path to watermark image file')
    parser.add_argument('--watermark-opacity', type=float, default=0.5, help='Watermark opacity (0.0-1.0, default: 0.5)')
    parser.add_argument('--watermark-mode', choices=['fit', 'fill'], default='fit', 
                       help='Watermark scaling mode: fit (entire image visible) or fill (cover entire frame, may crop) (default: fit)')
    parser.add_argument('--test-mode', action='store_true', help='Test mode: create only first 3 seconds of video')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return
    
    # Validate watermark opacity
    if args.watermark_opacity < 0.0 or args.watermark_opacity > 1.0:
        print(f"Error: Watermark opacity must be between 0.0 and 1.0 (got {args.watermark_opacity})")
        return
    
    # Generate file paths
    input_path = Path(args.input_file)
    
    # Determine lyrics CSV path
    if args.lyrics:
        lyrics_csv = Path(args.lyrics)
    else:
        lyrics_csv = input_path.parent / f"{input_path.stem}-歌詞.csv"
    
    # Check if lyrics CSV exists
    if not lyrics_csv.exists():
        print(f"Error: Lyrics CSV file '{lyrics_csv}' not found")
        print("Please run lyrics_matcher.py first")
        return
    
    # Read lyrics data
    print(f"Reading lyrics from: {lyrics_csv}")
    lyrics_data = read_lyrics_csv(lyrics_csv)
    
    if not lyrics_data:
        print("Error: No lyrics with timing information found in CSV")
        return
    
    print(f"Found {len(lyrics_data)} lyrics with timing information")
    
    # Determine title (default to filename without extension)
    title = args.title if args.title else input_path.stem
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        if args.mode == 'A':
            if args.test_mode:
                output_path = input_path.parent / f"{input_path.stem}-test.mp4"
            else:
                output_path = input_path.with_suffix('.mp4')
        else:  # Mode B
            if args.test_mode:
                output_path = input_path.parent / f"{input_path.stem}-lyrics-test.mp4"
            else:
                output_path = input_path.parent / f"{input_path.stem}-lyrics.mp4"
    
    # Execute based on mode
    if args.mode == 'A':
        print("Mode A: Creating video from MP3 and lyrics")
        create_video_with_lyrics_mode_a(
            input_path,
            lyrics_data,
            output_path,
            video_width=args.width,
            video_height=args.height,
            title=title,
            artist=args.artist,
            shader_path=args.shader,
            filter_type=args.filter,
            lyric_lines=args.lyric_lines,
            lyric_position=args.lyric_position,
            watermark_path=args.watermark,
            watermark_opacity=args.watermark_opacity,
            watermark_mode=args.watermark_mode,
            test_mode=args.test_mode
        )
    else:  # Mode B
        print("Mode B: Compositing lyrics onto existing video")
        create_video_with_lyrics_mode_b(
            input_path,
            lyrics_data,
            output_path,
            title=title,
            artist=args.artist,
            filter_type=args.filter,
            lyric_lines=args.lyric_lines,
            lyric_position=args.lyric_position,
            watermark_path=args.watermark,
            watermark_opacity=args.watermark_opacity,
            watermark_mode=args.watermark_mode,
            test_mode=args.test_mode
        )
    
    print(f"Video created successfully: {output_path}")


if __name__ == '__main__':
    main()