# Jimaku Subtitle Downloader

A Python tool for downloading Japanese anime subtitles from [Jimaku.cc](https://jimaku.cc) with intelligent episode detection and batch processing capabilities.

## Features

- 🔍 **Smart Search**: Search by anime title or AniList ID
- 📦 **Batch Downloads**: Download entire series automatically
- 🎯 **Intelligent Episode Detection**: Advanced pattern matching for episode numbers
- 🏆 **Quality Scoring**: Automatically selects the best subtitle files
- 📊 **Bulk Mode**: Download subtitles for top popular anime
- 🔄 **Resume Support**: Skip already downloaded series
- 📝 **Metadata Integration**: Fetches anime info from AniList

## Prerequisites

- Python 3.10+
- Jimaku.cc API key ([Get one here](https://jimaku.cc/api))
- patool and compatible extraction tools (7zip, unrar, etc.)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/k-jar/jimaku-downloader.git
cd jimaku-downloader
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API key:
```bash
# Create a .env file
echo "JIMAKU_API_KEY=your_api_key_here" > .env
```

4. Install extraction tools:
```bash
# On Ubuntu/Debian
sudo apt install p7zip-full unrar

# On macOS
brew install p7zip
brew install --cask rar

# On Windows
Download and install 7-Zip from https://www.7-zip.org/
```

## Usage

### Interactive Mode (Single Series)

Download a single anime series interactively:

```bash
python fetch_from_jimaku.py
```

Or specify a search query:

```bash
python fetch_from_jimaku.py --query "Steins Gate"
```

### Bulk Download Mode

Download subtitles for multiple popular anime:

```bash
# Download top 10 most popular anime
python bulk_download.py --limit 10

# Download with offset (e.g., anime ranked 50-100)
python bulk_download.py --limit 50 --offset 50

# Force re-download existing series
python bulk_download.py --limit 10 --force
```

## How It Works

### Episode Detection

The tool uses sophisticated pattern matching to identify episode numbers from subtitle filenames:

- Standard formats: `S01E01`, `EP01`, `Episode 1`
- Seasonal numbering: `Season 2 - 03`
- Bracket notation: `[01]`, `(12)`
- And many more patterns...

### Quality Scoring

Files are automatically scored based on:
- **Format preference**: Archives > SRT > ASS/SSA
- **Source quality**: Official releases (Netflix, etc.) get priority
- **Language purity**: Japanese-only subtitles preferred
- **Penalties**: OCR subtitles, mixed languages, closed captions

### Smart Windowing

The tool automatically handles different numbering schemes:
- Detects whether episodes use relative (1-12) or absolute (50-62) numbering
- Finds the most complete episode range
- Remaps episodes to standardized filenames (01.srt, 02.srt, etc.)

## Output Structure

Downloaded subtitles are organized as follows:

```
data/
└── raw_subtitles/
    ├── 12345_Steins Gate/
    │   ├── metadata.json
    │   ├── 01.srt
    │   ├── 02.srt
    │   ├── ...
    │   └── .completed
    └── logs/
        └── bulk_download_2025-01-15_14-30-00.log
```

## Configuration

### Environment Variables

- `JIMAKU_API_KEY`: Your Jimaku.cc API key (required)

### Constants (in fetch_from_jimaku.py)

```python
SERIES_EPISODE_LIMIT = 200      # Max episodes per series
MAX_SINGLE_FILE_SIZE = 1048576  # Max file size (1MB)
```

## API Rate Limiting

The tool includes built-in rate limiting:
- 1.5 second delay between series downloads
- 1 second delay between AniList API pages
- Respects Jimaku.cc API limits

## Troubleshooting

### "No extraction tool found"
Install extraction tools (see Installation section)

### "JIMAKU_API_KEY not found"
Create a `.env` file with your API key

### "Failed to extract archive"
Ensure you have the correct extraction tool installed for the archive format

### Episodes not detected
The filename pattern may not be recognized. Check the console output for the detected pattern.

## Acknowledgments

- [Jimaku.cc](https://jimaku.cc) for providing subtitle API access
- [AniList](https://anilist.co) for anime metadata
- Community subtitle contributors