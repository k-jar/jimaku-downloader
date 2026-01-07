# Jimaku Subtitle Downloader

CLI tool to batch download Japanese subtitles from Jimaku.cc. It handles search, episode parsing, and extracting archives automatically.

## Key Features

*   **Search**: Query by title or specific AniList ID.
*   **Batching**: Download entire series or top-ranking anime lists in bulk.
*   **Parsing**: Regex-based episode detection (handles `S01E01`, absolute numbering, and bracket formats).
*   **Filtering**: Prioritizes official sources (Netflix/Web-DL) and cleaner file formats (.srt) over OCR or ASS.
*   **Metadata**: Grabs show info from AniList to keep folders organized.

## Requirements

*   Python 3.10+
*   Jimaku.cc API key ([Get it here](https://jimaku.cc/api))
*   **System Tools**: `patool` requires 7zip and/or unrar installed on your system path.

## Setup

1.  **Clone and install:**
    ```bash
    git clone https://github.com/k-jar/jimaku-downloader.git
    cd jimaku-downloader
    pip install -r requirements.txt
    ```

2.  **Configure API Key:**
    Create a `.env` file in the root directory:
    ```bash
    echo "JIMAKU_API_KEY=your_key_here" > .env
    ```

3.  **Install System Extractors:**
    *   **Ubuntu/Debian**: `sudo apt install p7zip-full unrar`
    *   **macOS**: `brew install p7zip`
    *   **Windows**: Install [7-Zip](https://www.7-zip.org/) and ensure it's in your PATH.

## Usage

**Single Series**
Run the script to search interactively, or pass a query argument.
```bash
python fetch_from_jimaku.py --query "Steins Gate"
```

**Bulk Download**
Download top-rated anime from AniList.
```bash
# Top 10 anime
python bulk_download.py --limit 10

# Rank 50-100
python bulk_download.py --limit 50 --offset 50
```

## How Logic Works

**Episode Matching**
The script uses regex to identify episode numbers in filenames. It attempts to handle standard numbering (01-12) as well as absolute numbering (e.g., One Piece 1000+).

**Scoring System**
When multiple subtitle files exist for an episode, the script picks the best one based on:
1.  **Source**: Official Web-DL/Netflix > Fansubs > OCR.
2.  **Format**: Text-based (SRT) is preferred over ASS for compatibility.
3.  **Content**: Penalties are applied for files containing English text or closed captions.

**File Output**
Files are extracted, renamed to standard `01.srt`, `02.srt` format, and saved to `data/raw_subtitles/[ID]_[Title]/`.

#### **Note about AniList API Rate Limit**
The AniList API was in a degraded state at time of development. The rate limit was reduced from **90 requests/minute** to **30 requests/minute**. The script includes delays to respect this limit, so bulk downloads will be slower than usual.