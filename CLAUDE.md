# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a music video maker application that synchronizes lyrics with audio tracks using speech recognition technology.

### Core Functionality
- Reads audio from MP3 files
- Loads lyrics from text files
- Uses OpenAI Whisper for speech recognition to detect when lyrics are sung
- Generates timestamps for each lyric line
- Exports synchronized lyrics with timestamps as CSV

## Project Status

**Current State**: Initial setup phase - no source code implemented yet

## Development Setup

Since this is a Python project using Whisper, the following setup will be needed:

### Dependencies to Install
```bash
pip install openai-whisper
pip install pydub
pip install pandas
```

### Project Structure (Recommended)
```
tuidra-musicvideo-maker/
├── src/
│   ├── audio_processor.py    # MP3 file handling
│   ├── lyrics_parser.py      # Text file parsing
│   ├── whisper_sync.py       # Whisper integration
│   └── csv_exporter.py       # CSV output
├── tests/                    # Unit tests
├── data/                     # Sample MP3 and lyrics files
├── output/                   # Generated CSV files
├── requirements.txt          # Python dependencies
└── main.py                   # Main application entry point
```

## Common Commands

### Running the Application
```bash
python main.py --audio input.mp3 --lyrics lyrics.txt --output output.csv
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Running Tests
```bash
python -m pytest tests/
```

## Technical Considerations

1. **Whisper Model Selection**: Consider using different Whisper model sizes based on accuracy vs performance needs
2. **Audio Format Support**: Currently targeting MP3, but pydub supports multiple formats
3. **CSV Output Format**: Should include columns for timestamp, lyric text, and confidence scores
4. **Memory Management**: Large audio files may require streaming processing

## 作業の進捗をToDo.mdで管理する
- ToDo.mdを順番に処理し、終わったものに x チェックをつける。
