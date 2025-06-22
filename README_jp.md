# 音声認識とシェーダーで作る自動歌詞同期ミュージックビデオメーカー

## 概要

MP3ファイルから自動で歌詞を同期させたミュージックビデオを生成するPythonベースのツールです。OpenAI Whisperによる高精度な音声認識と、GLSLシェーダーによるリアルタイム背景生成を組み合わせることで、歌詞付き動画を簡単に作成できます。

## 主な特徴

### 🎵 音声認識による自動歌詞同期
- OpenAI Whisperを使用した高精度な日本語音声認識
- 認識結果とユーザー提供の歌詞を自動マッチング
- ローマ字変換による音韻的なマッチング精度の向上

### 🎨 リアルタイムシェーダー背景
- GLSL シェーダーによる動的な背景生成
- wgpu-shadertoyによるGPUアクセラレーション

### 🎬 柔軟な動画合成
- 既存動画への歌詞オーバーレイ（モードB）
- シェーダー背景での新規動画作成（モードA）
- ウォーターマーク、タイトル表示対応

## 技術的な実装詳細

### 1. 音声認識パイプライン

```python
# audio_recognition.py
def recognize_audio(mp3_path, model_size='large'):
    """
    Whisperを使用してMP3ファイルから音声を認識
    """
    model = whisper.load_model(model_size)
    result = model.transcribe(
        mp3_path,
        language='ja',
        verbose=True
    )
    
    # タイムスタンプ付きセグメントを抽出
    segments = []
    for segment in result['segments']:
        segments.append({
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text'].strip()
        })
    
    return segments
```

Whisperの各モデルサイズによる精度と処理速度のトレードオフ：
- `tiny`: 高速だが精度は低め（39M パラメータ）
- `base`: バランス型（74M）
- `small`: 日本語に推奨（244M）
- `medium`: 高精度（769M）
- `large`: 最高精度（1550M）

### 2. 歌詞マッチングアルゴリズム

歌詞マッチングは3段階のアプローチで実装：

```python
# lyrics_matcher.py - マルチステージマッチング
def match_lyrics_to_recognition(lyrics, recognition_results):
    """
    3段階のマッチングアルゴリズム：
    1. 完全一致
    2. ローマ字変換後の音韻マッチング
    3. 部分一致によるファジーマッチング
    """
    
    # Stage 1: 完全一致
    exact_matches = find_exact_matches(lyrics, recognition_results)
    
    # Stage 2: ローマ字マッチング（pykakasi使用）
    romaji_lyrics = [to_romaji(line) for line in lyrics]
    romaji_recognition = [to_romaji(seg['text']) for seg in recognition_results]
    phonetic_matches = find_phonetic_matches(romaji_lyrics, romaji_recognition)
    
    # Stage 3: 順序制約付き部分マッチング
    partial_matches = find_ordered_partial_matches(
        unmatched_lyrics, 
        unmatched_recognition,
        order_constraint=True
    )
    
    return merge_matches(exact_matches, phonetic_matches, partial_matches)
```

重要なポイント：
- **順序制約**: 歌詞の前後関係を保持してマッチング
- **音韻的類似度**: ローマ字変換により「こんにちは」と「konnichiwa」のような表記揺れを吸収
- **補間処理**: マッチしなかった歌詞は前後の歌詞から時間を補間

### 3. GLSLシェーダー統合

wgpu-shadertoyを使用したリアルタイムシェーダー背景生成：

```python
# shaderbg.py
class ShaderBackground:
    def __init__(self, shader_path, width, height):
        self.shader_code = self.load_shader(shader_path)
        self.renderer = ShadertoyRenderer(self.shader_code, width, height)
    
    def generate_frame(self, time):
        """
        指定時刻のシェーダーフレームを生成
        GPUで計算されるため高速
        """
        uniforms = {
            'iTime': time,
            'iResolution': [self.width, self.height, 1.0],
            'iMouse': [0.0, 0.0, 0.0, 0.0]
        }
        return self.renderer.render(uniforms)
```

### 4. 動画合成パイプライン

MoviePyを使用した効率的な動画生成：

```python
# video_generator.py
def create_music_video(audio_path, lyrics_data, options):
    """
    歌詞付きミュージックビデオを生成
    """
    # オーディオトラックを読み込み
    audio = AudioFileClip(audio_path)
    
    # 背景クリップを作成（シェーダーまたは既存動画）
    if options.mode == 'A':
        background = create_shader_background(options.shader, audio.duration)
    else:
        background = VideoFileClip(options.video_path)
    
    # 歌詞オーバーレイを生成
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
    
    # 全要素を合成
    final_video = CompositeVideoClip([background] + lyrics_clips)
    final_video.audio = audio
    
    return final_video
```

## 使用方法

### 基本的な使い方

```bash
# シンプルな実行
./make-mv-simple.sh audio.mp3 lyrics.txt watermark.png

# 詳細なオプション指定
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

### インストール

```bash
# 依存関係のインストール
pip install -r requirements.txt

# 初期セットアップ
./init.sh
```

### 必要な依存関係

- Python 3.8+
- OpenAI Whisper
- MoviePy
- wgpu-shadertoy
- pykakasi（日本語処理）
- pandas, numpy
- PIL/Pillow

## 実装のポイント

### 1. 日本語フォントの自動検出
```python
def find_japanese_font():
    """システムから日本語フォントを自動検出"""
    font_candidates = [
        'NotoSansCJK-Regular.ttc',
        'YuGothic.ttc',
        'Hiragino Sans GB.ttc'
    ]
    # フォントパスを探索...
```

### 2. 歌詞表示の最適化
- 文字数に応じた自動フォントサイズ調整
- 複数行表示時の透明度グラデーション
- 画面端での自動改行処理

### 3. エラーハンドリング
- Whisper認識失敗時のフォールバック
- シェーダーコンパイルエラーの検出と報告
- 不正な歌詞フォーマットの自動修正

## まとめ

このプロジェクトは、最新の音声認識技術とGPUシェーダーを組み合わせることで、従来は手作業で行っていた歌詞同期作業を自動化します。特に日本語楽曲に対して最適化されており、ローマ字変換を活用した高精度なマッチングが特徴です。

シェーダーによる動的な背景生成により、単調になりがちな歌詞動画に視覚的な魅力を加えることができ、YouTubeやSNS向けのコンテンツ制作に最適です。

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容について議論してください。