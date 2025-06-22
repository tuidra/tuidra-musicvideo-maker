#!/usr/bin/env python3
import argparse
import os
import whisper
import pandas as pd
from pathlib import Path


def transcribe_audio(audio_path, model_name='large'):
    """
    Transcribe audio file using Whisper and return segments with timestamps
    """
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    
    print(f"Transcribing: {audio_path}")
    result = model.transcribe(audio_path, language='ja', verbose=True)
    
    return result['segments']


def save_to_csv(segments, output_path):
    """
    Save transcription segments to CSV file
    """
    data = []
    for segment in segments:
        data.append({
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text'].strip()
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved transcription to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Audio recognition using Whisper')
    parser.add_argument('audio_file', help='Path to MP3 audio file')
    parser.add_argument('--model', default='large', 
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model to use (default: large)')
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found")
        return
    
    # Generate output filename
    audio_path = Path(args.audio_file)
    output_filename = f"{audio_path.stem}-whisper.csv"
    output_path = audio_path.parent / output_filename
    
    # Transcribe audio
    segments = transcribe_audio(args.audio_file, args.model)
    
    # Save to CSV
    save_to_csv(segments, output_path)


if __name__ == '__main__':
    main()