#!/usr/bin/env python3
import argparse
import os
import pandas as pd
from pathlib import Path
import difflib
from pykakasi import kakasi
import random


def to_romaji(text):
    """
    Convert Japanese text to romaji (romanized Japanese)
    """
    kks = kakasi()
    result = kks.convert(text)
    return ''.join([item['hepburn'] for item in result]).lower()


def read_whisper_results(csv_path):
    """
    Read Whisper transcription results from CSV
    """
    df = pd.read_csv(csv_path)
    return df


def read_lyrics(lyrics_path):
    """
    Read lyrics from text file
    """
    with open(lyrics_path, 'r', encoding='utf-8') as f:
        lyrics = [line.strip() for line in f.readlines() if line.strip()]
    return lyrics


def compute_all_similarities(whisper_data, lyrics_romaji):
    """
    Compute similarity matrix between all lyrics and whisper segments
    """
    similarity_matrix = []
    
    for lyric_idx, lyric_romaji in enumerate(lyrics_romaji):
        similarities = []
        for whisper_idx, whisper_item in enumerate(whisper_data):
            ratio = difflib.SequenceMatcher(None, lyric_romaji, whisper_item['romaji']).ratio()
            # Square the ratio to emphasize high similarity values
            weighted_ratio = ratio ** 2
            similarities.append({
                'lyric_idx': lyric_idx,
                'whisper_idx': whisper_idx,
                'similarity': weighted_ratio,
                'original_ratio': ratio
            })
        similarity_matrix.extend(similarities)
    
    return similarity_matrix


def check_order_constraint(new_match, existing_matches):
    """
    Check if a new match violates the order constraint with existing matches
    Returns True if the order is valid, False otherwise
    """
    lyric_idx = new_match['lyric_idx']
    whisper_idx = new_match['whisper_idx']
    
    for match in existing_matches:
        if match['lyric_idx'] < lyric_idx and match['whisper_idx'] >= whisper_idx:
            # A previous lyric is matched to a later whisper segment - violation
            return False
        if match['lyric_idx'] > lyric_idx and match['whisper_idx'] <= whisper_idx:
            # A later lyric is matched to an earlier whisper segment - violation
            return False
    
    return True


def multi_stage_matching(whisper_data, lyrics_romaji, lyrics):
    """
    Multi-stage matching algorithm that fixes high-confidence matches first
    """
    # Compute all similarities
    all_similarities = compute_all_similarities(whisper_data, lyrics_romaji)
    
    # Sort by similarity (highest first)
    all_similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Initialize results
    matched_results = [{
        'start': '',
        'end': '',
        'lyric': lyrics[i],
        'whisper_text': '',
        'similarity': 0,
        'lyric_romaji': lyrics_romaji[i],
        'whisper_romaji': ''
    } for i in range(len(lyrics))]
    
    used_lyric_indices = set()
    used_whisper_indices = set()
    existing_matches = []
    
    # Define similarity thresholds (squared values)
    thresholds = [0.9**2, 0.8**2, 0.7**2, 0.6**2, 0.5**2, 0.4**2, 0.3**2]
    
    for threshold in thresholds:
        print(f"  Processing matches with similarity >= {threshold**0.5:.1f}")
        
        # Find matches at this threshold level
        for sim in all_similarities:
            if sim['similarity'] < threshold:
                break  # Since sorted, no more matches at this threshold
            
            lyric_idx = sim['lyric_idx']
            whisper_idx = sim['whisper_idx']
            
            # Skip if already matched
            if lyric_idx in used_lyric_indices or whisper_idx in used_whisper_indices:
                continue
            
            # Check order constraint
            if not check_order_constraint(sim, existing_matches):
                continue
            
            # Valid match - add it
            whisper_item = whisper_data[whisper_idx]
            matched_results[lyric_idx] = {
                'start': whisper_item['start'],
                'end': whisper_item['end'],
                'lyric': lyrics[lyric_idx],
                'whisper_text': whisper_item['text'],
                'similarity': sim['original_ratio'],
                'lyric_romaji': lyrics_romaji[lyric_idx],
                'whisper_romaji': whisper_item['romaji']
            }
            
            used_lyric_indices.add(lyric_idx)
            used_whisper_indices.add(whisper_idx)
            existing_matches.append({
                'lyric_idx': lyric_idx,
                'whisper_idx': whisper_idx
            })
        
        # Sort existing matches by lyric index for easier constraint checking
        existing_matches.sort(key=lambda x: x['lyric_idx'])
    
    return matched_results


def fill_remaining_gaps(matched_results, whisper_data, lyrics_romaji):
    """
    Fill remaining gaps in the matched results by interpolating timing
    """
    # Find consecutive unmatched lyrics and try to assign them whisper segments
    # that fit within the time constraints of surrounding matches
    
    i = 0
    while i < len(matched_results):
        if matched_results[i]['start'] == '':
            # Found unmatched lyric, find the extent of the gap
            gap_start = i
            while i < len(matched_results) and matched_results[i]['start'] == '':
                i += 1
            gap_end = i
            
            # Find surrounding matched lyrics
            prev_match_idx = gap_start - 1 if gap_start > 0 else None
            next_match_idx = gap_end if gap_end < len(matched_results) else None
            
            # Try to interpolate timing
            if prev_match_idx is not None and next_match_idx is not None:
                # Gap is between two matches
                prev_end_time = float(matched_results[prev_match_idx]['end'])
                next_start_time = float(matched_results[next_match_idx]['start'])
                
                # Distribute the time evenly among unmatched lyrics
                gap_size = gap_end - gap_start
                time_per_lyric = (next_start_time - prev_end_time) / (gap_size + 1)
                
                for j, idx in enumerate(range(gap_start, gap_end)):
                    start_time = prev_end_time + (j + 1) * time_per_lyric * 0.8
                    end_time = prev_end_time + (j + 1) * time_per_lyric * 1.2
                    
                    matched_results[idx]['start'] = f"{start_time:.3f}"
                    matched_results[idx]['end'] = f"{end_time:.3f}"
                    matched_results[idx]['whisper_text'] = '[interpolated]'
                    matched_results[idx]['similarity'] = 0.0
        else:
            i += 1
    
    return matched_results


def match_lyrics_single_pass(whisper_data, lyrics_romaji, lyrics, threshold=0.4, lookahead=5, use_multi_stage=True):
    """
    Single pass of matching lyrics with whisper data
    """
    if use_multi_stage:
        # Use multi-stage matching algorithm
        return multi_stage_matching(whisper_data, lyrics_romaji, lyrics)
    
    # Fallback to original sequential matching
    matched_results = []
    whisper_index = 0
    
    # Process each lyric line
    for lyric_idx, lyric in enumerate(lyrics):
        lyric_romaji = lyrics_romaji[lyric_idx]
        
        # Find best matching whisper segment from current position onward
        best_match = None
        best_ratio = 0
        best_index = whisper_index
        
        # Look ahead in whisper results
        for i in range(whisper_index, min(whisper_index + lookahead, len(whisper_data))):
            # Compare using romaji for better phonetic matching
            ratio = difflib.SequenceMatcher(None, lyric_romaji, whisper_data[i]['romaji']).ratio()
            # Square the ratio to emphasize high values
            weighted_ratio = ratio ** 2
            if weighted_ratio > best_ratio:
                best_ratio = weighted_ratio
                best_match = whisper_data[i]
                best_index = i
        
        # If good match found, use it (compare squared values)
        if best_ratio > threshold ** 2 and best_match:
            matched_results.append({
                'start': best_match['start'],
                'end': best_match['end'],
                'lyric': lyric,
                'whisper_text': best_match['text'],
                'similarity': best_ratio ** 0.5,  # Convert back to original scale for display
                'lyric_romaji': lyric_romaji,
                'whisper_romaji': best_match['romaji']
            })
            whisper_index = best_index + 1
        else:
            # No good match found, still output the lyric with empty timing
            matched_results.append({
                'start': '',
                'end': '',
                'lyric': lyric,
                'whisper_text': '',
                'similarity': 0,
                'lyric_romaji': lyric_romaji,
                'whisper_romaji': ''
            })
    
    return matched_results


def count_matched_lyrics(results):
    """
    Count how many lyrics have timing information
    """
    return sum(1 for r in results if r['start'] != '')


def merge_unmatched_lyrics(results):
    """
    Merge unmatched lyrics with the previous matched lyric
    This ensures all lyrics have timing information
    """
    merged_results = []
    last_matched_index = -1
    
    for i, result in enumerate(results):
        if result['start'] != '':
            # This lyric has timing, keep it
            merged_results.append(result)
            last_matched_index = len(merged_results) - 1
        else:
            # This lyric has no timing
            if last_matched_index >= 0:
                # Merge with previous matched lyric
                prev_result = merged_results[last_matched_index]
                prev_result['lyric'] += ' ' + result['lyric']
                prev_result['lyric_romaji'] += ' ' + result['lyric_romaji']
                print(f"Merged unmatched lyric '{result['lyric']}' with previous")
            else:
                # No previous match, keep as is (will have empty timing)
                merged_results.append(result)
    
    return merged_results


def match_lyrics(whisper_df, lyrics, max_attempts=5):
    """
    Match Whisper transcription with actual lyrics using multiple attempts
    All lyrics must be output - lyrics are the baseline
    """
    # Convert to romaji for better matching
    lyrics_romaji = [to_romaji(lyric) for lyric in lyrics]
    whisper_data = []
    for _, row in whisper_df.iterrows():
        whisper_data.append({
            'text': row['text'],
            'romaji': to_romaji(row['text']),
            'start': row['start'],
            'end': row['end']
        })
    
    best_results = None
    best_match_count = 0
    
    # Try multiple matching attempts with different parameters
    for attempt in range(max_attempts):
        # Vary parameters for each attempt
        if attempt == 0:
            # First attempt: multi-stage matching
            threshold = 0.4
            lookahead = 5
            use_multi_stage = True
        elif attempt < max_attempts // 2:
            # First few attempts: multi-stage with different random seeds
            threshold = 0.3 + random.uniform(0, 0.3)
            lookahead = random.randint(3, 8)
            use_multi_stage = True
            # Shuffle similarity matrix slightly for variation
            random.seed(attempt)
        else:
            # Later attempts: try sequential matching as fallback
            threshold = 0.3 + random.uniform(0, 0.3)
            lookahead = random.randint(3, 8)
            use_multi_stage = False
        
        # Perform matching
        results = match_lyrics_single_pass(
            whisper_data, lyrics_romaji, lyrics, 
            threshold=threshold, lookahead=lookahead, use_multi_stage=use_multi_stage
        )
        
        # Count matched lyrics
        match_count = count_matched_lyrics(results)
        
        method_str = " (multi-stage)" if use_multi_stage else " (sequential)"
        print(f"Attempt {attempt + 1}/{max_attempts}: Matched {match_count}/{len(lyrics)} lyrics{method_str}")
        
        # Keep best result
        if match_count > best_match_count:
            best_match_count = match_count
            best_results = results
    
    print(f"\nBest result: {best_match_count}/{len(lyrics)} lyrics matched")
    
    # Try to fill remaining gaps with interpolation
    if best_match_count < len(lyrics):
        print("\nFilling gaps with interpolated timing...")
        best_results = fill_remaining_gaps(best_results, whisper_data, lyrics_romaji)
        final_match_count = count_matched_lyrics(best_results)
        print(f"After gap filling: {final_match_count}/{len(best_results)} lyrics have timing")
        
        # If still have unmatched, try merging
        if final_match_count < len(best_results):
            print("\nMerging remaining unmatched lyrics with previous matched lyrics...")
            best_results = merge_unmatched_lyrics(best_results)
            final_match_count = count_matched_lyrics(best_results)
            print(f"After merging: {final_match_count}/{len(best_results)} lyrics have timing")
    
    return best_results


def save_matched_results(matched_results, output_path):
    """
    Save matched results to CSV
    """
    df = pd.DataFrame(matched_results)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved matched lyrics to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Match Whisper transcription with lyrics')
    parser.add_argument('audio_file', help='Path to MP3 audio file')
    parser.add_argument('--lyrics', default=None, 
                       help='Path to lyrics text file (default: audio_file.txt)')
    parser.add_argument('--max-attempts', type=int, default=500,
                       help='Maximum number of matching attempts (default: 5)')
    
    args = parser.parse_args()
    
    # Generate file paths
    audio_path = Path(args.audio_file)
    whisper_csv = audio_path.parent / f"{audio_path.stem}-whisper.csv"
    
    # Check if Whisper CSV exists
    if not whisper_csv.exists():
        print(f"Error: Whisper results file '{whisper_csv}' not found")
        print("Please run audio_recognition.py first")
        return
    
    # Determine lyrics file path
    if args.lyrics:
        lyrics_path = Path(args.lyrics)
    else:
        lyrics_path = audio_path.with_suffix('.txt')
    
    # Check if lyrics file exists
    if not lyrics_path.exists():
        print(f"Error: Lyrics file '{lyrics_path}' not found")
        return
    
    # Read data
    print(f"Reading Whisper results from: {whisper_csv}")
    whisper_df = read_whisper_results(whisper_csv)
    
    print(f"Reading lyrics from: {lyrics_path}")
    lyrics = read_lyrics(lyrics_path)
    
    # Match lyrics
    print(f"Matching lyrics with max {args.max_attempts} attempts...")
    matched_results = match_lyrics(whisper_df, lyrics, max_attempts=args.max_attempts)
    
    # Save results
    output_path = audio_path.parent / f"{audio_path.stem}-歌詞.csv"
    save_matched_results(matched_results, output_path)


if __name__ == '__main__':
    main()