"""
Jimaku Bulk Subtitle Downloader
Downloads subtitles for multiple popular anime from Jimaku.cc automatically.

Usage:
    python bulk_download.py --limit 10
    python bulk_download.py --limit 50 --offset 100
    python bulk_download.py --limit 10 --force
"""

import os
import sys
import requests
import argparse
import time
import json
from datetime import datetime

from fetch_from_jimaku import (
    search_anime_jimaku,
    download_series_by_id,
    fetch_anilist_metadata,
    RAW_SUBTITLES_DIR,
    SCRIPT_DIR,
)


# ==================== CONFIGURATION ====================

LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


# ==================== LOGGING ====================

class DualLogger:
    """Redirects output to both console and log file simultaneously."""
    
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ==================== LOCAL STATE MANAGEMENT ====================

def get_existing_anilist_ids() -> set:
    """
    Scan local subtitle directories for completed downloads.
    Only returns IDs for series marked with '.completed' file.
    
    Returns:
        Set of AniList IDs that have been completed
    """
    existing_ids = set()
    
    if not os.path.exists(RAW_SUBTITLES_DIR):
        return existing_ids

    for folder in os.listdir(RAW_SUBTITLES_DIR):
        folder_path = os.path.join(RAW_SUBTITLES_DIR, folder)
        meta_path = os.path.join(folder_path, "metadata.json")
        marker_path = os.path.join(folder_path, ".completed")

        # Both metadata AND completion marker must exist
        if os.path.exists(meta_path) and os.path.exists(marker_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("anilist_id"):
                        existing_ids.add(data["anilist_id"])
            except Exception:
                pass
                
    return existing_ids


# ==================== ANILIST API ====================

def fetch_top_anime(limit: int = 1000, offset: int = 0) -> list:
    """
    Fetch top popular anime from AniList with pagination.
    
    Args:
        limit: Total number of anime to fetch
        offset: Starting rank (0-indexed)
        
    Returns:
        List of anime metadata dictionaries
    """
    url = "https://graphql.anilist.co"
    
    query = """
    query ($page: Int, $perPage: Int) {
      Page (page: $page, perPage: $perPage) {
        pageInfo {
          hasNextPage
        }
        media (type: ANIME, sort: POPULARITY_DESC, format_in: [TV, MOVIE]) {
          id
          title {
            romaji
            native
            english
          }
          episodes
          format
        }
      }
    }
    """
    
    all_anime = []
    per_page = 50  # AniList's max per request
    current_page = (offset // per_page) + 1
    initial_skip = offset % per_page

    while len(all_anime) < limit:
        variables = {
            "page": current_page,
            "perPage": per_page
        }

        try:
            response = requests.post(url, json={"query": query, "variables": variables})
            response.raise_for_status()
            data = response.json()["data"]["Page"]
            
            new_media = data["media"]
            
            # Skip initial offset on first page
            if initial_skip > 0:
                new_media = new_media[initial_skip:]
                initial_skip = 0
            
            all_anime.extend(new_media)
            
            print(f"  Fetched page {current_page} ({len(all_anime)}/{limit} items)")

            # Stop if no more pages
            if not data["pageInfo"]["hasNextPage"]:
                break
                
            current_page += 1
            time.sleep(1)  # Rate limiting

        except Exception as e:
            print(f"✗ Failed to fetch AniList page {current_page}: {e}")
            break
    
    # Trim to exact limit
    return all_anime[:limit]


# ==================== BULK DOWNLOAD LOGIC ====================

def run_bulk_download(limit: int, offset: int = 0, force: bool = False):
    """
    Main bulk download orchestrator.
    
    Args:
        limit: Number of anime to process
        offset: Starting rank (0-indexed)
        force: Re-download even if already completed
    """
    print("="*70)
    print(f"BULK DOWNLOAD: {limit} anime starting at rank #{offset + 1}")
    print("="*70)
    
    # Fetch popular anime from AniList
    print(f"\nFetching anime list from AniList...")
    top_anime = fetch_top_anime(limit, offset)
    
    if not top_anime:
        print("✗ Failed to fetch anime list")
        return

    # Get already downloaded series
    existing_ids = get_existing_anilist_ids()
    print(f"\n✓ Found {len(existing_ids)} series already downloaded locally")
    print("="*70)

    # Process each anime
    for i, anime in enumerate(top_anime):
        anilist_id = anime["id"]
        title_romaji = anime["title"]["romaji"]
        title_native = anime["title"]["native"]

        print(f"\n[{i+1}/{limit}] Processing: {title_romaji}")
        print(f"  Native: {title_native}")
        print(f"  AniList ID: {anilist_id}")

        # Skip if already downloaded (unless force mode)
        if not force and anilist_id in existing_ids:
            print(f"  ✓ Already downloaded - skipping")
            continue

        # Search Jimaku by AniList ID
        results = search_anime_jimaku(anilist_id=anilist_id)

        # Fallback: Search by title
        if not results:
            print(f"  → No results for ID, trying name search...")
            results = search_anime_jimaku(query=title_romaji)

        if not results:
            print(f"  ✗ No entries found on Jimaku")
            continue

        # Match result to AniList ID
        target_entry = None
        for res in results:
            if res.get("anilist_id") == anilist_id:
                target_entry = res
                break

        # Fallback: Use first result if no exact match
        if not target_entry and results:
            target_entry = results[0]
            print(f"  → Fuzzy match: Using '{target_entry.get('name', 'Unknown')}'")

        if not target_entry:
            print(f"  ✗ Could not match entry to AniList ID")
            continue

        # Enrich metadata
        metadata = fetch_anilist_metadata(anilist_id) or {}
        metadata.update({
            "anilist_id": anilist_id,
            "jimaku_id": target_entry["id"],
            "episodes": anime["episodes"],
        })

        # Determine title for directory name
        jimaku_title = (
            target_entry.get("title_jp") or 
            metadata.get("title_jp") or 
            target_entry.get("name") or 
            metadata.get("title_romaji") or
            f"Series_{anilist_id}"
        )
        
        is_movie = anime["format"] == "MOVIE"

        # Download
        print(f"  → Starting download...")
        success = download_series_by_id(
            entry_id=target_entry["id"],
            series_title=jimaku_title,
            metadata=metadata,
            is_movie=is_movie
        )

        if success:
            existing_ids.add(anilist_id)
            print(f"  ✓ Successfully completed")
        else:
            print(f"  ⚠ Download incomplete")

        # Rate limiting
        time.sleep(1.5)

    print("\n" + "="*70)
    print("BULK DOWNLOAD COMPLETE")
    print("="*70)


# ==================== CLI INTERFACE ====================

def main():
    """Main entry point for bulk download."""
    parser = argparse.ArgumentParser(
        description="Bulk download anime subtitles by AniList popularity ranking"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of popular anime to download (default: 10)"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting rank, 0-indexed (default: 0 = #1 most popular)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download series even if already completed"
    )

    args = parser.parse_args()

    # Setup logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(LOGS_DIR, f"bulk_download_{timestamp}.log")
    
    # Redirect stdout/stderr to log file AND console
    logger = DualLogger(log_file)
    sys.stdout = logger
    sys.stderr = logger
    
    print(f"Logging to: {log_file}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        run_bulk_download(
            limit=args.limit,
            offset=args.offset,
            force=args.force
        )
    except KeyboardInterrupt:
        print("\n\n✗ Download interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\nEnded at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.close()


if __name__ == "__main__":
    main()