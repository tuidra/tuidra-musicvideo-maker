#!/usr/bin/env python3
"""
Background image filter
Process background images with various effects
"""
import argparse
import os
import numpy as np
from pathlib import Path
from PIL import Image


def apply_interlace_filter(image_array, line_color=(0, 0, 0), line_height=1, spacing=2):
    """
    Apply interlace effect by drawing black lines
    
    Args:
        image_array: numpy array (H, W, 3) representing RGB image
        line_color: RGB tuple for line color (default: black)
        line_height: Height of each line in pixels
        spacing: Spacing between lines in pixels
    
    Returns:
        numpy array with interlace effect applied
    """
    height, width, channels = image_array.shape
    result = image_array.copy()
    
    # Draw horizontal lines at regular intervals
    y = 0
    while y < height:
        # Draw line
        for line_y in range(y, min(y + line_height, height)):
            result[line_y, :] = line_color
        y += line_height + spacing
    
    return result


def apply_blur_filter(image_array, blur_radius=2):
    """
    Apply blur filter using PIL
    
    Args:
        image_array: numpy array (H, W, 3) representing RGB image
        blur_radius: Blur radius in pixels
    
    Returns:
        numpy array with blur applied
    """
    from PIL import ImageFilter
    
    img = Image.fromarray(image_array.astype(np.uint8), 'RGB')
    blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return np.array(blurred)


def apply_brightness_filter(image_array, brightness=1.0):
    """
    Apply brightness adjustment
    
    Args:
        image_array: numpy array (H, W, 3) representing RGB image
        brightness: Brightness multiplier (1.0 = no change, >1.0 = brighter, <1.0 = darker)
    
    Returns:
        numpy array with brightness adjusted
    """
    result = image_array.astype(np.float32) * brightness
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_contrast_filter(image_array, contrast=1.0):
    """
    Apply contrast adjustment
    
    Args:
        image_array: numpy array (H, W, 3) representing RGB image
        contrast: Contrast multiplier (1.0 = no change, >1.0 = more contrast, <1.0 = less contrast)
    
    Returns:
        numpy array with contrast adjusted
    """
    # Convert to float and center around 0
    result = image_array.astype(np.float32) - 128.0
    # Apply contrast
    result *= contrast
    # Add back the center and clip
    result += 128.0
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_scanlines_filter(image_array, line_opacity=0.3, line_spacing=2):
    """
    Apply scanlines effect (darker lines across the image)
    
    Args:
        image_array: numpy array (H, W, 3) representing RGB image
        line_opacity: Opacity of the scan lines (0.0 = invisible, 1.0 = completely black)
        line_spacing: Spacing between scan lines
    
    Returns:
        numpy array with scanlines effect applied
    """
    height, width, channels = image_array.shape
    result = image_array.copy().astype(np.float32)
    
    # Create scanlines pattern
    for y in range(0, height, line_spacing):
        result[y, :] *= (1.0 - line_opacity)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def process_frame(input_path, output_path, filters):
    """
    Process a single frame with specified filters
    
    Args:
        input_path: Path to input image
        output_path: Path to output image
        filters: List of filter configurations
    """
    try:
        # Load image
        img = Image.open(input_path)
        img_array = np.array(img)
        
        # Apply filters in sequence
        for filter_config in filters:
            filter_type = filter_config['type']
            filter_params = filter_config.get('params', {})
            
            if filter_type == 'interlace':
                img_array = apply_interlace_filter(img_array, **filter_params)
            elif filter_type == 'blur':
                img_array = apply_blur_filter(img_array, **filter_params)
            elif filter_type == 'brightness':
                img_array = apply_brightness_filter(img_array, **filter_params)
            elif filter_type == 'contrast':
                img_array = apply_contrast_filter(img_array, **filter_params)
            elif filter_type == 'scanlines':
                img_array = apply_scanlines_filter(img_array, **filter_params)
            else:
                print(f"Warning: Unknown filter type '{filter_type}'")
        
        # Save processed image
        result_img = Image.fromarray(img_array, 'RGB')
        result_img.save(output_path, 'JPEG', quality=85)
        
    except Exception as e:
        print(f"Error processing frame {input_path}: {e}")


def process_frames(input_dir, output_dir=None, filters=None):
    """
    Process all frames in a directory
    
    Args:
        input_dir: Directory containing input frames
        output_dir: Directory for output frames (if None, overwrites input)
        filters: List of filter configurations
    
    Returns:
        List of output frame paths
    """
    if filters is None:
        # Default: interlace filter
        filters = [{'type': 'interlace', 'params': {'line_height': 1, 'spacing': 2}}]
    
    # Create output directory if needed
    if output_dir is None:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files
    input_path = Path(input_dir)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(input_path.glob(ext))
    
    image_files.sort()
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return []
    
    print(f"Processing {len(image_files)} frames with {len(filters)} filters...")
    
    output_paths = []
    for i, input_file in enumerate(image_files):
        if output_dir == input_dir:
            output_file = input_file  # Overwrite
        else:
            output_file = Path(output_dir) / input_file.name
        
        process_frame(str(input_file), str(output_file), filters)
        output_paths.append(str(output_file))
        
        if i % 100 == 0:
            print(f"Processed frame {i+1}/{len(image_files)}")
    
    print(f"Frame processing complete: {len(output_paths)} frames")
    return output_paths


def main():
    parser = argparse.ArgumentParser(description='Filter background images')
    parser.add_argument('input_dir', help='Directory containing input frames')
    parser.add_argument('--output', help='Output directory (default: overwrite input)')
    parser.add_argument('--filter', choices=['interlace', 'blur', 'brightness', 'contrast', 'scanlines'], 
                       default='interlace', help='Filter type (default: interlace)')
    
    # Filter-specific parameters
    parser.add_argument('--line-height', type=int, default=1, help='Interlace line height (default: 1)')
    parser.add_argument('--spacing', type=int, default=2, help='Interlace spacing (default: 2)')
    parser.add_argument('--blur-radius', type=float, default=2.0, help='Blur radius (default: 2.0)')
    parser.add_argument('--brightness', type=float, default=1.0, help='Brightness multiplier (default: 1.0)')
    parser.add_argument('--contrast', type=float, default=1.0, help='Contrast multiplier (default: 1.0)')
    parser.add_argument('--line-opacity', type=float, default=0.3, help='Scanlines opacity (default: 0.3)')
    parser.add_argument('--line-spacing', type=int, default=2, help='Scanlines spacing (default: 2)')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found")
        return
    
    # Prepare filter configuration
    if args.filter == 'interlace':
        filters = [{
            'type': 'interlace',
            'params': {
                'line_height': args.line_height,
                'spacing': args.spacing
            }
        }]
    elif args.filter == 'blur':
        filters = [{
            'type': 'blur',
            'params': {
                'blur_radius': args.blur_radius
            }
        }]
    elif args.filter == 'brightness':
        filters = [{
            'type': 'brightness',
            'params': {
                'brightness': args.brightness
            }
        }]
    elif args.filter == 'contrast':
        filters = [{
            'type': 'contrast',
            'params': {
                'contrast': args.contrast
            }
        }]
    elif args.filter == 'scanlines':
        filters = [{
            'type': 'scanlines',
            'params': {
                'line_opacity': args.line_opacity,
                'line_spacing': args.line_spacing
            }
        }]
    
    # Process frames
    output_paths = process_frames(args.input_dir, args.output, filters)
    
    if output_paths:
        print(f"Processed {len(output_paths)} frames")
        if args.output:
            print(f"Output saved in: {args.output}")
        else:
            print(f"Input frames updated in place: {args.input_dir}")
    else:
        print("No frames were processed")


if __name__ == '__main__':
    main()