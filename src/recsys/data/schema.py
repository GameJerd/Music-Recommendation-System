"""Expected layout of the Kaggle 'vatsalmavani/spotify-dataset' files (Spotify 1921-2020)."""

TRACKS_FILE = "data.csv"
ARTISTS_FILE = "data_w_genres.csv"
REQUIRED_FILES = [TRACKS_FILE, ARTISTS_FILE]

AUDIO_COLUMNS = [
    "danceability", "energy", "valence", "acousticness", "instrumentalness",
    "liveness", "speechiness", "loudness", "tempo",
]

EXPECTED_COLUMNS = {
    TRACKS_FILE: set(AUDIO_COLUMNS) | {
        "id", "name", "artists", "year", "release_date", "popularity",
        "explicit", "duration_ms", "key", "mode",
    },
    ARTISTS_FILE: set(AUDIO_COLUMNS) | {
        "artists", "genres", "popularity", "duration_ms", "key", "mode", "count",
    },
}
