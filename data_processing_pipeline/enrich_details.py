import pandas as pd
import requests
import time
import os
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
API_KEY = '719a37bb28bd0a34b4e34fdc2a3566e2'
OUTPUT_FILE = 'enriched_movie_descriptions.csv'
LINKS_FILE = './data/link.csv'
SAVE_INTERVAL = 50  # Save to disk every N movies
REQ_DELAY = 0.25    # Delay between requests to respect rate limits (4 req/sec)

def enrich_movies():
    # 1. Load the links file
    if not os.path.exists(LINKS_FILE):
        print(f"Error: {LINKS_FILE} not found.")
        return

    links_df = pd.read_csv(LINKS_FILE)
    links_df = links_df.dropna(subset=['tmdbId'])

    # 2. Load existing progress to support Resuming
    enriched_list = []
    processed_ids = set()
    
    if os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            processed_ids = set(existing_df['movieId'].unique())
            enriched_list = existing_df.to_dict('records')
            print(f"Found existing progress. Resuming from movie {len(processed_ids)}...")
        except Exception as e:
            print(f"Could not load existing file: {e}. Starting fresh.")

    # Filter out already processed movies
    to_process = links_df[~links_df['movieId'].isin(processed_ids)]
    
    if len(to_process) == 0:
        print("All movies already enriched!")
        return

    print(f"Processing {len(to_process)} remaining movies...")

    # 3. Main Loop
    try:
        # Use tqdm for a professional progress bar
        for i, (_, row) in enumerate(tqdm(to_process.iterrows(), total=len(to_process), desc="Enriching")):
            movie_id = int(row['movieId'])
            tmdb_id = int(row['tmdbId'])
            
            url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={API_KEY}&language=en-US"
            
            try:
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    enriched_list.append({
                        'movieId': movie_id,
                        'tmdbId': tmdb_id,
                        'title': data.get('title', 'Unknown'),
                        'description': data.get('overview', ''),
                        'poster': data.get('poster_path', '')
                    })
                elif response.status_code == 429:
                    # TMDB Rate Limit - usually shouldn't happen with 0.25s delay
                    print("\nRate limit hit. Waiting 10 seconds...")
                    time.sleep(10)
                elif response.status_code == 404:
                    print(f"\nMovie ID {tmdb_id} not found on TMDB. Skipping.")
                
                # Incremental Save
                if (i + 1) % SAVE_INTERVAL == 0:
                    pd.DataFrame(enriched_list).to_csv(OUTPUT_FILE, index=False)
                
                # Rate limit safety
                time.sleep(REQ_DELAY)
                
            except requests.exceptions.RequestException as e:
                print(f"\nConnection error for ID {tmdb_id}: {e}. Retrying in 5s...")
                time.sleep(5)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving current progress...")
    
    # 4. Final Save
    final_df = pd.DataFrame(enriched_list)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done! Total enriched movies: {len(final_df)}")

if __name__ == "__main__":
    enrich_movies()