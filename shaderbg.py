#!/usr/bin/env python3
"""
Shader background generator
Generates frames using GLSL shaders
"""
import argparse
import os
import numpy as np
import tempfile
from pathlib import Path

try:
    from wgpu_shadertoy import Shadertoy
    from PIL import Image
    SHADERTOY_AVAILABLE = True
except ImportError:
    SHADERTOY_AVAILABLE = False
    print("Warning: wgpu-shadertoy not available. Install with: pip install wgpu-shadertoy")


def generate_shader_frames(shader_path, video_width, video_height, duration, fps=24, output_dir=None, quality=85):
    """
    Generate frames using a shader file
    
    Args:
        shader_path: Path to GLSL shader file
        video_width: Video width in pixels
        video_height: Video height in pixels
        duration: Duration in seconds
        fps: Frames per second
        output_dir: Output directory (if None, creates temp directory)
        quality: JPEG quality (1-100)
    
    Returns:
        tuple: (frame_paths_list, temp_directory_path)
    """
    if not SHADERTOY_AVAILABLE:
        print("Error: wgpu-shadertoy not available")
        return [], None
    
    if not os.path.exists(shader_path):
        print(f"Error: Shader file not found: {shader_path}")
        return [], None
    
    try:
        # Read shader code
        with open(shader_path, 'r') as f:
            shader_code = f.read()
        
        # Create shadertoy renderer with offscreen rendering
        print(f"Creating shader renderer: {video_width}x{video_height}")
        renderer = Shadertoy(shader_code, resolution=(video_width, video_height), offscreen=True)
        
        # Create output directory
        if output_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="shader_frames_")
            print(f"Using temporary directory: {temp_dir}")
        else:
            temp_dir = output_dir
            os.makedirs(temp_dir, exist_ok=True)
            print(f"Using output directory: {temp_dir}")
        
        # Generate frames
        total_frames = int(duration * fps)
        frame_paths = []
        
        print(f"Generating {total_frames} shader frames...")
        
        for frame_num in range(total_frames):
            time = frame_num / fps
            
            # Get frame as memoryview and convert to numpy array
            frame_data = renderer.snapshot(time_float=time, frame=frame_num)
            
            # Convert memoryview to numpy array
            frame_array = np.frombuffer(frame_data, dtype=np.uint8)
            frame_array = frame_array.reshape((video_height, video_width, 4))  # RGBA format
            
            # Convert RGBA to RGB for PIL
            frame_rgb = frame_array[:, :, :3]
            
            # Convert to PIL Image and save as JPEG
            img = Image.fromarray(frame_rgb, 'RGB')
            frame_path = os.path.join(temp_dir, f"frame_{frame_num:06d}.jpg")
            img.save(frame_path, 'JPEG', quality=quality)
            frame_paths.append(frame_path)
            
            if frame_num % 100 == 0:
                print(f"Generated frame {frame_num}/{total_frames}")
        
        print(f"Shader frame generation complete: {len(frame_paths)} frames")
        return frame_paths, temp_dir
        
    except Exception as e:
        print(f"Error generating shader frames: {e}")
        import traceback
        traceback.print_exc()
        return [], None


def cleanup_frames(frame_paths, temp_dir=None):
    """
    Clean up frame files and temporary directory
    
    Args:
        frame_paths: List of frame file paths
        temp_dir: Temporary directory path (if None, derived from frame paths)
    """
    if not frame_paths:
        return
    
    # Get the temporary directory from the first frame path if not provided
    if temp_dir is None:
        temp_dir = os.path.dirname(frame_paths[0])
    
    # Remove all frame files
    for frame_path in frame_paths:
        try:
            os.remove(frame_path)
        except:
            pass
    
    # Remove the temporary directory if it's empty
    try:
        os.rmdir(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")
    except:
        pass


def main():
    parser = argparse.ArgumentParser(description='Generate shader background frames')
    parser.add_argument('shader_file', help='Path to GLSL shader file')
    parser.add_argument('--width', type=int, default=1920, help='Video width (default: 1920)')
    parser.add_argument('--height', type=int, default=1080, help='Video height (default: 1080)')
    parser.add_argument('--duration', type=float, required=True, help='Duration in seconds')
    parser.add_argument('--fps', type=int, default=24, help='Frames per second (default: 24)')
    parser.add_argument('--output', help='Output directory (default: temporary directory)')
    parser.add_argument('--quality', type=int, default=85, help='JPEG quality 1-100 (default: 85)')
    parser.add_argument('--no-cleanup', action='store_true', help='Do not clean up frames after generation')
    
    args = parser.parse_args()
    
    # Check if shader file exists
    if not os.path.exists(args.shader_file):
        print(f"Error: Shader file '{args.shader_file}' not found")
        return
    
    # Generate frames
    frame_paths, temp_dir = generate_shader_frames(
        args.shader_file,
        args.width,
        args.height,
        args.duration,
        fps=args.fps,
        output_dir=args.output,
        quality=args.quality
    )
    
    if frame_paths:
        print(f"Generated {len(frame_paths)} frames")
        if args.output:
            print(f"Frames saved in: {args.output}")
        else:
            print(f"Frames saved in temporary directory: {temp_dir}")
            
        # Clean up if requested and using temporary directory
        if not args.no_cleanup and args.output is None:
            cleanup_frames(frame_paths, temp_dir)
    else:
        print("Frame generation failed")


if __name__ == '__main__':
    main()