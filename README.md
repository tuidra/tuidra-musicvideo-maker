# Automatic Lyrics Synchronization Music Video Maker with Voice Recognition and Shaders

## Overview

A Python-based tool that automatically generates music videos with synchronized lyrics from MP3 files. It combines high-precision voice recognition using OpenAI Whisper with real-time background generation using GLSL shaders to easily create lyric videos.

![ScreenShot](https://raw.githubusercontent.com/tuidra/tuidra-musicvideo-maker/refs/heads/main/screenshot.png)

## Key Features

### 🎵 Automatic Lyrics Synchronization with Voice Recognition
- High-precision Japanese voice recognition using OpenAI Whisper
- Automatic matching of recognition results with user-provided lyrics
- Improved phonetic matching accuracy through romanization conversion

### 🎨 Real-time Shader Backgrounds
- Dynamic background generation using GLSL shaders
- GPU acceleration with wgpu-shadertoy

### 🎬 Flexible Video Composition
- Lyrics overlay on existing videos (Mode B)
- New video creation with shader backgrounds (Mode A)
- Support for watermarks and title display

## Technical Implementation Details

### 1. Voice Recognition Pipeline

```python
# audio_recognition.py
def recognize_audio(mp3_path, model_size='large'):
    """
    Recognize audio from MP3 file using Whisper
    """
    model = whisper.load_model(model_size)
    result = model.transcribe(
        mp3_path,
        language='ja',
        verbose=True
    )
    
    # Extract segments with timestamps
    segments = []
    for segment in result['segments']:
        segments.append({
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text'].strip()
        })
    
    return segments
```

Trade-off between accuracy and processing speed for each Whisper model size:
- `tiny`: Fast but lower accuracy (39M parameters)
- `base`: Balanced (74M)
- `small`: Recommended for Japanese (244M)
- `medium`: High accuracy (769M)
- `large`: Highest accuracy (1550M)

### 2. Lyrics Matching Algorithm

Lyrics matching is implemented with a 3-stage approach:

```python
# lyrics_matcher.py - Multi-stage matching
def match_lyrics_to_recognition(lyrics, recognition_results):
    """
    3-stage matching algorithm:
    1. Exact match
    2. Phonetic matching after romanization conversion
    3. Fuzzy matching with partial matches
    """
    
    # Stage 1: Exact match
    exact_matches = find_exact_matches(lyrics, recognition_results)
    
    # Stage 2: Romanization matching (using pykakasi)
    romaji_lyrics = [to_romaji(line) for line in lyrics]
    romaji_recognition = [to_romaji(seg['text']) for seg in recognition_results]
    phonetic_matches = find_phonetic_matches(romaji_lyrics, romaji_recognition)
    
    # Stage 3: Partial matching with order constraints
    partial_matches = find_ordered_partial_matches(
        unmatched_lyrics, 
        unmatched_recognition,
        order_constraint=True
    )
    
    return merge_matches(exact_matches, phonetic_matches, partial_matches)
```

Key points:
- **Order constraints**: Maintain the sequential relationship of lyrics during matching
- **Phonetic similarity**: Absorb notation variations like "こんにちは" and "konnichiwa" through romanization conversion
- **Interpolation processing**: Interpolate timing for unmatched lyrics from surrounding lyrics

### 3. GLSL Shader Integration

Real-time shader background generation using wgpu-shadertoy:

```python
# shaderbg.py
class ShaderBackground:
    def __init__(self, shader_path, width, height):
        self.shader_code = self.load_shader(shader_path)
        self.renderer = ShadertoyRenderer(self.shader_code, width, height)
    
    def generate_frame(self, time):
        """
        Generate shader frame at specified time
        High-speed processing with GPU computation
        """
        uniforms = {
            'iTime': time,
            'iResolution': [self.width, self.height, 1.0],
            'iMouse': [0.0, 0.0, 0.0, 0.0]
        }
        return self.renderer.render(uniforms)
```

### 4. Video Composition Pipeline

Efficient video generation using MoviePy:

```python
# video_generator.py
def create_music_video(audio_path, lyrics_data, options):
    """
    Generate music video with lyrics
    """
    # Load audio track
    audio = AudioFileClip(audio_path)
    
    # Create background clip (shader or existing video)
    if options.mode == 'A':
        background = create_shader_background(options.shader, audio.duration)
    else:
        background = VideoFileClip(options.video_path)
    
    # Generate lyrics overlay
    lyrics_clips = []
    for lyric in lyrics_data:
        text_clip = TextClip(
            lyric['text'],
            fontsize=calculate_optimal_fontsize(lyric['text'], options.width),
            color='white',
            stroke_color='black',
            stroke_width=2,
            font=options.font
        ).set_position(('center', 'bottom')).set_duration(
            lyric['end'] - lyric['start']
        ).set_start(lyric['start'])
        
        lyrics_clips.append(text_clip)
    
    # Compose all elements
    final_video = CompositeVideoClip([background] + lyrics_clips)
    final_video.audio = audio
    
    return final_video
```

## Usage

### Basic Usage

```bash
# Simple execution
./make-mv-simple.sh audio.mp3 lyrics.txt watermark.png

# Detailed option specification
python video_generator.py input.mp3 \
  --mode A \
  --shader shader/animated_galaxy.glsl \
  --width 1920 \
  --height 1080 \
  --lyric-lines 3 \
  --lyric-position center \
  --watermark logo.png \
  --watermark-opacity 0.8
```

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Initial setup
./init.sh
```

### Required Dependencies

- Python 3.8+
- OpenAI Whisper
- MoviePy
- wgpu-shadertoy
- pykakasi (Japanese processing)
- pandas, numpy
- PIL/Pillow

## Implementation Highlights

### 1. Automatic Japanese Font Detection
```python
def find_japanese_font():
    """Automatically detect Japanese fonts from system"""
    font_candidates = [
        'NotoSansCJK-Regular.ttc',
        'YuGothic.ttc',
        'Hiragino Sans GB.ttc'
    ]
    # Search font paths...
```

### 2. Lyrics Display Optimization
- Automatic font size adjustment based on character count
- Transparency gradient for multi-line display
- Automatic line wrapping at screen edges

### 3. Error Handling
- Fallback for Whisper recognition failures
- Detection and reporting of shader compilation errors
- Automatic correction of invalid lyrics formats

## Summary

This project automates the traditionally manual lyrics synchronization process by combining cutting-edge voice recognition technology with GPU shaders. It is specifically optimized for Japanese music and features high-precision matching using romanization conversion.

The dynamic background generation using shaders adds visual appeal to otherwise monotonous lyric videos, making it ideal for content creation for YouTube and social media.

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss the proposed changes.