file# Bug Report - Song Recommender CLI

## Summary
Tested the CLI application (`main.py`) and identified multiple bugs related to missing genre data and encoding issues.

---

## Bugs Found

| # | Bug | Location | Description |
|---|-----|-----------|-------------|
| 1 | **Unicode Encoding Error** | `main.py:114` | Crashes when printing song names with non-ASCII chars (e.g., 'ő', 'á'). Error: `'charmap' codec can't encode character '\u0151'` |
| 2 | **Missing Genre Column** | `data_loader.py:12` | Dataset doesn't have 'genre' column - only has: id, name, album, album_id, artists, artist_ids, track_number, disc_number, explicit, and audio features |
| 3 | **Genre Features Broken** | `recommender.py:86-126` | All genre-based recommendations fail with "Genre 'X' not found" because genre column is missing |
| 4 | **Wrong Fuzzy Match** | `recommender.py:28-33` | Searching "blinding lights" incorrectly matches to "nightwings" - fuzzy matching threshold too low |
| 5 | **Artist Name Parsing Bug** | `data_loader.py` | Artist names stored as string lists (e.g., "['bricks']" instead of "bricks") - incorrect parsing of the artists column |
| 6 | **List Genres Shows Only "unknown"** | `main.py:136-145` | Since all songs default to "unknown" genre (due to missing genre column), listing genres shows only "unknown" |

---

## Root Cause
The dataset (`tracks_features.csv`) lacks a genre column. The code expects a `genre` column but the actual CSV only contains track metadata and audio features without genre information.

---

## Test Commands Used
```bash
# Test song recommendation
echo -e "1\nblinding lights\n5\n" | python main.py --rebuild --sample 1000

# Test artist recommendation
echo -e "2\nthe weeknd\n5\n" | python main.py --rebuild --sample 1000

# Test genre recommendation
echo -e "3\nrock\n5\n" | python main.py --rebuild --sample 1000

# Test list genres
echo -e "4\n5\n" | python main.py --rebuild --sample 1000
```
