"""
Jimaku Subtitle Downloader
Downloads Japanese anime subtitles from Jimaku.cc with intelligent episode detection.

Usage:
    python fetch_from_jimaku.py
    python fetch_from_jimaku.py --query "Anime Title"
"""

import json
import sys
import os
import requests
import re
import patoolib
import shutil
import tempfile
import argparse
from typing import List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# ==================== CONFIGURATION ====================

API_KEY = os.environ.get("JIMAKU_API_KEY")
BASE_URL = "https://jimaku.cc/api"

# Directory structure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RAW_SUBTITLES_DIR = os.path.join(DATA_DIR, "raw_subtitles")

# Limits
SERIES_EPISODE_LIMIT = 200  # Maximum episodes to download per series
MAX_SINGLE_FILE_SIZE = 1024 * 1024  # 1MB - skip bloated subtitle files

# Supported archive formats
ARCHIVE_EXTS = (".zip", ".rar", ".7z")

# Validate API key
if not API_KEY:
    print("Error: JIMAKU_API_KEY not found in environment variables.")
    print("Please create a .env file with: JIMAKU_API_KEY=your_api_key")
    sys.exit(1)

HEADERS = {"Authorization": API_KEY}


# ==================== EPISODE DETECTION PATTERNS ====================

# Regex patterns for extracting episode numbers from filenames
# Ordered by specificity to avoid false positives
EPISODE_PATTERNS = [
    # Season/Volume/Part explicit numbering: "Season 3 - 08", "S3 - 01"
    r"(?:[Ss]eason|[Vv]ol\.?|[Pp]art|[Ss])\s{0,2}\d{1,3}(?:[ _\-]|\s-\s)(\d{1,3})(?!\d)",
    # Standard Scene: S01E01, s1e1
    r"[Ss]\d{1,2}[Ee](\d{1,3})",
    # Explicit Episode: EP01, Ep.01, episode 1
    r"(?:[Ee][Pp]|[Ee]pisode)[\.]?\s?(\d{1,3})",
    # Separated Number: " - 01 - ", " - 23 ", " _01_"
    r"[ _\-](\d{1,3})(?:v\d)?[ _\-\[\(]",
    # Surrounded by brackets: [01], (01)
    r"[\[\(](\d{1,3})[\]\)](?![pP]|bit|x)",
    # Loose number at end of file: "Name - 01.ass"
    r"[ _\-](\d{1,3})\.(?:ass|srt|ssa|vtt)$",
    # Hash/x Marker: #01, x01 (exclude resolution markers like x264, x1080)
    r"(?<!\d)[#x](?!(?:264|265|266|1080|720))(\d{1,3})(?!\d)",
    # "E" marker preceded by separator: "NARUTO E01", "Title_E01"
    r"(?:^|[\s_\-\.])E(\d{1,3})(?!\d)",
]

# Language detection patterns
CN_PATTERN = re.compile(
    r"(?:^|[^a-z])(cn|ch|cht|chs|tc|sc|zh|big5|gb)(?:$|[^a-z])", re.IGNORECASE
)
JP_PATTERN = re.compile(r"(?:^|[^a-z])(ja|jp|jpn|japanese)(?:$|[^a-z])", re.IGNORECASE)
EN_PATTERN = re.compile(r"(?:^|[^a-z])(en|eng|english)(?:$|[^a-z])", re.IGNORECASE)


# ==================== FILE MANAGEMENT ====================

def save_files_to_local(
    candidates: List[dict], destination_dir: str, existing_episodes: set
) -> set:
    """
    Save subtitle files to local storage with standardized naming.
    
    Args:
        candidates: List of files with 'path', 'name', and 'remapped_ep'
        destination_dir: Target directory for saved files
        existing_episodes: Set of episode numbers already downloaded
        
    Returns:
        Set of episode numbers successfully saved
    """
    saved_eps = set()
    
    for target in candidates:
        ep_num = target["remapped_ep"]
        src_path = target["path"]

        if ep_num in existing_episodes:
            continue

        # Standardized filename: "01.ass", "12.srt"
        _, ext = os.path.splitext(target["name"])
        new_filename = f"{ep_num:02d}{ext}"
        dest_path = os.path.join(destination_dir, new_filename)

        try:
            shutil.copy2(src_path, dest_path)
            saved_eps.add(ep_num)
        except Exception as e:
            print(f"  ✗ Failed to save Episode {ep_num}: {e}")

    if saved_eps:
        print(f"  ✓ Saved {len(saved_eps)} new episodes: {sorted(list(saved_eps))}")

    return saved_eps


# ==================== EPISODE NUMBER EXTRACTION ====================

def extract_episode_number(filename: str) -> Optional[int]:
    """
    Extract episode number from filename using pattern matching.
    
    Args:
        filename: Name of the subtitle file
        
    Returns:
        Episode number if found, None otherwise
    """
    for pattern in EPISODE_PATTERNS:
        match = re.search(pattern, filename)
        if match:
            try:
                num = int(match.group(1))
                # Episode 0 exists, but 2000+ is likely a year
                if 0 <= num < 1900:
                    return num
            except ValueError:
                continue
    return None


# ==================== QUALITY SCORING ====================

def score_file_candidate(filename: str, is_movie: bool = False) -> int:
    """
    Assign a quality score to a subtitle file.
    Higher scores indicate better quality/preference.
    
    Args:
        filename: Name of the file to score
        is_movie: Whether this is a movie (affects archive scoring)
        
    Returns:
        Integer score (negative = reject, positive = prefer)
    """
    score = 0
    name_lower = filename.lower()

    # Base format scores
    if name_lower.endswith(".srt"):
        score += 10
    elif name_lower.endswith((".ass", ".ssa", ".vtt")):
        score += 5
    elif name_lower.endswith(ARCHIVE_EXTS):
        if is_movie:
            score = -20  # Archives rarely needed for single movies
        else:
            score += 15  # Archives great for series
    else:
        return -1000  # Reject unknown formats

    # Quality indicators
    if any(x in name_lower for x in ["netflix", "nf", "webrip", "web-dl", "official"]):
        score += 5

    # Penalize OCR (often contains character recognition errors)
    if "ocr" in name_lower:
        score -= 5

    # Penalize CC (closed captions include sound effects)
    if "cc" in name_lower:
        score -= 2

    # Language preference scoring
    has_jp = bool(JP_PATTERN.search(name_lower))
    has_cn = bool(CN_PATTERN.search(name_lower))
    has_en = bool(EN_PATTERN.search(name_lower))

    if has_jp:
        if has_cn:
            score -= 1000  # JP/CN mix is problematic for analysis
        elif has_en:
            score -= 10  # JP/EN mix is annoying but manageable
        else:
            score += 20  # Pure Japanese - ideal
    elif has_cn:
        score -= 1000  # Chinese only - not useful
    elif has_en:
        score -= 20  # English only - not useful
    else:
        score += 20  # Likely Japanese only

    return score


# ==================== EPISODE SELECTION ====================

def select_best_candidates(
    candidates: List[dict], total_episodes: Optional[int], max_limit: int = 200
) -> List[dict]:
    """
    Select the best subtitle files by finding the most complete episode window.
    Handles both relative (1-12) and absolute (25-36) numbering schemes.
    
    Args:
        candidates: List of candidate files with 'ep' and 'score'
        total_episodes: Expected episode count (from AniList)
        max_limit: Maximum episodes to download
        
    Returns:
        List of best candidates with 'remapped_ep' field added
    """
    if not candidates:
        return []

    # Group files by detected episode number
    ep_map = {}
    for c in candidates:
        ep = c["ep"]
        if ep not in ep_map:
            ep_map[ep] = []
        ep_map[ep].append(c)

    sorted_eps = sorted(ep_map.keys())
    
    # Determine target window size
    target_count = (
        total_episodes
        if (total_episodes and total_episodes > 0)
        else SERIES_EPISODE_LIMIT
    )
    window_size = min(target_count, max_limit)

    # Find the best starting episode number
    # Each start point is evaluated for completeness
    best_start_ep = sorted_eps[0]
    best_score = -float('inf')

    for start_ep in sorted_eps:
        found_count = 0
        total_quality = 0
        
        for i in range(window_size):
            target = start_ep + i
            if target in ep_map:
                found_count += 1
                best_file_score = max(f["score"] for f in ep_map[target])
                total_quality += best_file_score
        
        # Prioritize completeness over quality
        window_score = (found_count * 100) + total_quality
        
        if window_score > best_score:
            best_score = window_score
            best_start_ep = start_ep

    # Build final results
    results = []
    print(f"  → Selected numbering scheme starting at {best_start_ep} (window: {window_size})")

    for i in range(window_size):
        target_label = best_start_ep + i

        if target_label in ep_map:
            # Pick highest scoring file for this episode
            best_file = max(ep_map[target_label], key=lambda x: x["score"])
            final_file = best_file.copy()
            
            # Remap to sequential numbering (1, 2, 3...)
            final_file["remapped_ep"] = i + 1
            results.append(final_file)

    return results


# ==================== API INTERACTIONS ====================

def search_anime_jimaku(
    query: Optional[str] = None, anilist_id: Optional[int] = None
) -> List[dict]:
    """
    Search Jimaku for anime by title or AniList ID.
    
    Args:
        query: Anime title to search for
        anilist_id: AniList ID to search for
        
    Returns:
        List of search results
    """
    print(f"Searching Jimaku (Query: {query}, AniList ID: {anilist_id})...")
    params = {"anime": "true"}

    if anilist_id:
        params["anilist_id"] = anilist_id
    elif query:
        params["query"] = query
    else:
        return []

    try:
        res = requests.get(
            f"{BASE_URL}/entries/search",
            params=params,
            headers=HEADERS,
        )
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print(f"✗ Jimaku search failed: {e}")
        return []


def get_entry_files(entry_id: int) -> List[dict]:
    """
    Fetch all subtitle files for a Jimaku entry.
    
    Args:
        entry_id: Jimaku entry ID
        
    Returns:
        List of file metadata dictionaries
    """
    try:
        res = requests.get(f"{BASE_URL}/entries/{entry_id}/files", headers=HEADERS)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print(f"✗ Failed to get files: {e}")
        return []


def download_file(url: str, dest_path: str) -> str:
    """
    Download a file from URL to local path.
    
    Args:
        url: File URL
        dest_path: Local destination path
        
    Returns:
        Path to downloaded file
    """
    print(f"  → Downloading {os.path.basename(dest_path)}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return dest_path


def fetch_anilist_metadata(anilist_id: int) -> Optional[dict]:
    """
    Fetch anime metadata from AniList API.
    
    Args:
        anilist_id: AniList anime ID
        
    Returns:
        Dictionary of metadata or None if failed
    """
    if not anilist_id:
        return None

    query = """
    query ($id: Int) {
      Media (id: $id, type: ANIME) {
        title {
            native
            english
            romaji
        }
        description(asHtml: false)
        averageScore
        popularity
        genres
        episodes
        coverImage {
          extraLarge
        }
      }
    }
    """
    url = "https://graphql.anilist.co"
    
    try:
        response = requests.post(
            url, json={"query": query, "variables": {"id": anilist_id}}
        )
        data = response.json()
        media = data["data"]["Media"]

        return {
            "title_jp": media["title"]["native"],
            "title_en": media["title"]["english"],
            "title_romaji": media["title"]["romaji"],
            "anilist_rating": media["averageScore"],
            "popularity": media["popularity"],
            "thumbnail_url": media["coverImage"]["extraLarge"],
            "description": media["description"],
            "genres": media["genres"],
            "episodes": media["episodes"],
        }
    except Exception as e:
        print(f"✗ AniList fetch error: {e}")
        return None


# ==================== DIRECTORY SCANNING ====================

def scan_directory_for_candidates(
    extract_dir: str, metadata: dict, is_movie: bool = False
) -> List[dict]:
    """
    Scan extracted directory for subtitle files and select best candidates.
    
    Args:
        extract_dir: Directory to scan
        metadata: Anime metadata
        is_movie: Whether this is a movie
        
    Returns:
        List of best candidate files
    """
    all_candidates = []
    total_episodes = (
        metadata.get("episodes", SERIES_EPISODE_LIMIT)
        if metadata
        else SERIES_EPISODE_LIMIT
    )

    print("  → Scanning extracted files...")
    for root, _, filenames in os.walk(extract_dir):
        for filename in filenames:
            full_path = os.path.join(root, filename)
            
            # Skip oversized files
            try:
                file_size = os.path.getsize(full_path)
                if file_size > MAX_SINGLE_FILE_SIZE:
                    continue
            except OSError:
                continue

            ep_num = extract_episode_number(filename)
            
            # For movies without episode number, assume episode 1
            if ep_num is None and is_movie:
                if score_file_candidate(filename, is_movie=True) > -100:
                    ep_num = 1

            if ep_num is not None:
                score = score_file_candidate(filename, is_movie=is_movie)
                if score > -100:  # Filter out junk files
                    all_candidates.append({
                        "path": full_path,
                        "name": filename,
                        "ep": ep_num,
                        "score": score,
                    })

    # Apply windowing logic to select best episodes
    targets = select_best_candidates(all_candidates, total_episodes)
    return targets


# ==================== DOWNLOAD STRATEGY ====================

def generate_download_attempts(
    files: List[dict],
    is_movie: bool = False,
    total_episodes: Optional[int] = 12,
    max_limit: int = SERIES_EPISODE_LIMIT,
) -> List[Tuple[str, List[dict]]]:
    """
    Generate prioritized download strategies.
    Tries archives first (batch), then individual files.
    
    Args:
        files: List of available files from Jimaku
        is_movie: Whether this is a movie
        total_episodes: Expected episode count
        max_limit: Maximum episodes to attempt
        
    Returns:
        List of (strategy_type, files) tuples
    """
    attempts = []
    archives = []
    subs = []

    for f in files:
        name_lower = f["name"].lower()
        is_archive = name_lower.endswith(ARCHIVE_EXTS)
        file_size = f.get("size", 0)

        # Skip oversized individual subtitle files
        if not is_archive and file_size > MAX_SINGLE_FILE_SIZE:
            continue

        score = score_file_candidate(f["name"], is_movie=is_movie)
        f["score"] = score

        if score > 0:
            if is_archive:
                archives.append(f)
            else:
                subs.append(f)

    # Priority 1: Archives (sorted by score and size)
    archives.sort(key=lambda x: (x["score"], x["size"]), reverse=True)
    for archive in archives:
        attempts.append(("batch", [archive]))

    # Priority 2: Individual files (as fallback)
    candidates = []
    for s in subs:
        ep = extract_episode_number(s["name"])
        if ep is None and is_movie:
            ep = 1

        if ep is not None:
            s["ep"] = ep
            candidates.append(s)

    best_individuals = select_best_candidates(
        candidates, total_episodes, max_limit=max_limit
    )

    if best_individuals:
        avg_score = sum(s["score"] for s in best_individuals) / len(best_individuals)
        if avg_score > -10:
            attempts.append(("individual", best_individuals))

    return attempts


# ==================== SERIES SETUP ====================

def setup_series_directory(
    entry_id: int, series_title: str, metadata: dict
) -> Tuple[str, bool]:
    """
    Create series directory and check if already complete.
    
    Args:
        entry_id: Jimaku entry ID
        series_title: Series title for directory name
        metadata: Anime metadata to save
        
    Returns:
        Tuple of (directory_path, is_already_complete)
    """
    # Sanitize title for filesystem
    safe_title = "".join(
        c for c in series_title if c.isalnum() or c in (" ", "_", "-")
    ).strip()
    series_dir = os.path.join(RAW_SUBTITLES_DIR, f"{entry_id}_{safe_title}")

    # Check if already completed
    if os.path.exists(os.path.join(series_dir, ".completed")):
        return series_dir, True

    os.makedirs(series_dir, exist_ok=True)

    # Save metadata
    with open(os.path.join(series_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return series_dir, False


# ==================== FILE PROCESSING ====================

def process_downloaded_file(
    local_path: str,
    target_file: dict,
    strategy: str,
    temp_dir: str,
    metadata: dict,
    is_movie: bool,
) -> List[dict]:
    """
    Process a downloaded file (extract archive or prepare individual file).
    
    Args:
        local_path: Path to downloaded file
        target_file: File metadata
        strategy: 'batch' or 'individual'
        temp_dir: Temporary directory for extraction
        metadata: Anime metadata
        is_movie: Whether this is a movie
        
    Returns:
        List of candidate subtitle files
    """
    candidates = []

    if strategy == "batch":
        extract_path = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_path, exist_ok=True)
        try:
            patoolib.extract_archive(local_path, outdir=extract_path)
            candidates = scan_directory_for_candidates(
                extract_path, metadata, is_movie=is_movie
            )
        except Exception as e:
            print(f"  ✗ Archive extraction failed: {e}")

    elif strategy == "individual":
        ep_num = target_file.get("remapped_ep")
        if ep_num:
            candidates = [{
                "path": local_path,
                "name": target_file["name"],
                "remapped_ep": ep_num
            }]

    return candidates


# ==================== MAIN DOWNLOAD FUNCTION ====================

def download_series_by_id(
    entry_id: int, series_title: str, metadata: dict, is_movie: bool = False
) -> bool:
    """
    Download all subtitles for a series using intelligent strategy selection.
    
    Args:
        entry_id: Jimaku entry ID
        series_title: Series title
        metadata: Anime metadata
        is_movie: Whether this is a movie
        
    Returns:
        True if download completed successfully
    """
    # Calculate episode limits
    total_episodes = metadata.get("episodes")
    if not total_episodes:
        total_episodes = SERIES_EPISODE_LIMIT
    target_count = min(total_episodes, SERIES_EPISODE_LIMIT)

    # Setup storage directory
    series_dir, is_complete = setup_series_directory(
        entry_id=entry_id, series_title=series_title, metadata=metadata
    )
    
    if is_complete:
        print(f"✓ Skipping {series_title}: Already complete")
        return True

    # Get available files
    files = get_entry_files(entry_id)
    if not files:
        print(f"✗ No files found on Jimaku for {series_title}")
        return False

    # Generate download strategies
    attempts = generate_download_attempts(
        files,
        is_movie=is_movie,
        total_episodes=target_count,
        max_limit=SERIES_EPISODE_LIMIT,
    )

    if not attempts:
        print(f"✗ No suitable files found for {series_title}")
        return False

    print(f"Found {len(files)} files. Generated {len(attempts)} download strategies.")
    print(f"Target: ~{target_count} episodes")

    # Execute download strategies
    downloaded_episodes = set()

    for i, (strategy, download_targets) in enumerate(attempts):
        if len(downloaded_episodes) >= target_count:
            print("✓ Target episode count reached")
            break

        print(f"\n[Attempt {i+1}/{len(attempts)}] Strategy: {strategy.upper()}")
        print(f"  Files: {len(download_targets)}")

        with tempfile.TemporaryDirectory() as temp_dir:
            for target_file in download_targets:
                # Skip if already have this episode
                if strategy == "individual":
                    if target_file.get("remapped_ep") in downloaded_episodes:
                        continue

                dest = os.path.join(temp_dir, target_file["name"])

                try:
                    download_file(target_file["url"], dest)

                    # Process file
                    candidates = process_downloaded_file(
                        dest, target_file, strategy, temp_dir, metadata, is_movie
                    )

                    # Save results
                    new_eps = save_files_to_local(
                        candidates, series_dir, existing_episodes=downloaded_episodes
                    )
                    downloaded_episodes.update(new_eps)

                except Exception as e:
                    print(f"  ✗ Download failed: {e}")

    # Mark as complete if we got enough episodes
    if (
        len(downloaded_episodes) >= total_episodes
        or len(downloaded_episodes) >= SERIES_EPISODE_LIMIT
    ):
        with open(os.path.join(series_dir, ".completed"), "w") as f:
            f.write("done")
        print(f"\n✓ {series_title} marked as complete ({len(downloaded_episodes)} episodes)")
        return True
    else:
        print(f"\n⚠ Downloaded {len(downloaded_episodes)} episodes but expected {total_episodes}")
        return False


# ==================== CLI INTERFACE ====================

def main():
    """Main entry point for interactive subtitle download."""
    parser = argparse.ArgumentParser(
        description="Download Japanese anime subtitles from Jimaku.cc"
    )
    parser.add_argument(
        "--query",
        help="Anime title to search for (optional, will prompt if not provided)"
    )
    args = parser.parse_args()

    # Ensure data directory exists
    os.makedirs(RAW_SUBTITLES_DIR, exist_ok=True)

    # Get search query
    query = args.query or input("Enter anime title to search: ")
    results = search_anime_jimaku(query)

    if not results:
        print("✗ No results found")
        return

    # Display results
    print("\n" + "="*60)
    print("Search Results:")
    print("="*60)
    for i, entry in enumerate(results):
        title_jp = entry.get("title_jp", entry.get("name", "Unknown"))
        print(f"{i+1}. {title_jp} (Jimaku ID: {entry['id']})")
    print("="*60)

    # User selection
    try:
        choice = int(input("\nSelect number (or 0 to cancel): ")) - 1
        if choice < 0:
            print("Cancelled")
            return
        selected = results[choice]
    except (ValueError, IndexError):
        print("✗ Invalid selection")
        return

    jimaku_id = selected["id"]
    anilist_id = selected.get("anilist_id")

    # Fetch metadata
    metadata = fetch_anilist_metadata(anilist_id) or {}
    metadata["anilist_id"] = anilist_id
    metadata["jimaku_id"] = jimaku_id

    flags = selected.get("flags", {})
    is_movie = flags.get("movie", False)
    total_episodes = metadata.get("episodes", SERIES_EPISODE_LIMIT)

    title_jp = selected.get("title_jp", selected.get("name", f"Series_{jimaku_id}"))
    
    print(f"\n{'='*60}")
    print(f"Selected: {title_jp}")
    print(f"Type: {'Movie' if is_movie else 'TV Series'}")
    if total_episodes:
        print(f"Episodes: {total_episodes}")
    print("="*60)

    # Download
    download_series_by_id(
        entry_id=jimaku_id,
        series_title=title_jp,
        metadata=metadata,
        is_movie=is_movie
    )


if __name__ == "__main__":
    main()