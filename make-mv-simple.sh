#!/bin/bash

FILE=$1
LYRICS=$2
WATERMARK=$3
SHADER=""

if [ -z "$FILE" ]; then
    echo "Usage: $0 <file> <lyrics> <watermark>"
    exit 1
fi

if [ -n "$LYRICS" ]; then
    LYRICS="--lyrics $LYRICS"
fi

if [ -n "$WATERMARK" ]; then
    WATERMARK="--watermark $WATERMARK --watermark-opacity 0.7 --watermark-mode fill"
fi

python audio_recognition.py "$FILE"
python lyrics_matcher.py "$FILE" $LYRICS
python video_generator.py "$FILE" --width 960 --height 512 --artist Tuidra $SHADER $WATERMARK #--test-mode