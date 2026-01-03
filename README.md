# Video Frame Diff Extractor

映像ファイルからフレーム差分を検出し、変化があったフレームを画像として抽出するCLIツールです。

## 機能

- 動画ファイル（MP4, AVI, MOV, MKV, WebM）からフレームを読み込み
- 連続するフレーム間の差分を検出
- 変化が検出されたフレームをPNG画像として保存
- クロップ画像による条件付き抽出（オプション）
- 走査エリアの指定（オプション）

## 環境構築

### 必要条件

- Python 3.8以上

### セットアップ

1. リポジトリをクローン

```bash
git clone https://github.com/uniSIF/video-frame-diff-extractor.git
cd video-frame-diff-extractor
```

2. 仮想環境を作成・有効化

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# Windows の場合: venv\Scripts\activate
```

3. pipをアップグレード（推奨）

```bash
pip install --upgrade pip
```

4. パッケージをインストール

```bash
pip install -e .
```

## 使い方

### 基本的な使い方

```bash
python -m src.main 動画ファイル.mp4
```

出力ファイルは `./output/` ディレクトリに保存されます。

### オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `-o, --output-dir` | 出力ディレクトリのパス | `./output` |
| `-t, --threshold` | 差分検出閾値（0.0-1.0） | `0.05` |
| `--crop-image` | クロップ判定条件画像のパス | なし |
| `--scan-area` | 走査エリア（x,y,width,height形式） | なし |

### 実行例

```bash
# 基本的な使い方
python -m src.main video.mp4

# 出力先を指定
python -m src.main video.mp4 -o ./frames

# 閾値を調整（値を大きくすると検出感度が下がる）
python -m src.main video.mp4 -t 0.1

# 出力先と閾値を両方指定
python -m src.main video.mp4 -o ./frames -t 0.08

# クロップ画像を指定
python -m src.main video.mp4 --crop-image pattern.png

# 走査エリアを指定（x=100, y=200, 幅=300, 高さ=400）
python -m src.main video.mp4 --scan-area 100,200,300,400
```

### 出力ファイル

出力ファイルは以下の形式で保存されます：

```
frame_NNNNNN_MMmSSs.png
```

- `NNNNNN`: フレーム番号（6桁、ゼロ埋め）
- `MM`: 分
- `SS`: 秒

例: `frame_000375_00m07s.png` は375フレーム目（0分7秒時点）の画像

## 開発

### テストの実行

```bash
pip install -e ".[dev]"
pytest
```

## ライセンス

MIT License
