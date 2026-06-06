from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR = Path(__file__).resolve().parent / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Popular movies for realistic testing
movies_data = [
    {'title': 'The Shawshank Redemption', 'release_date': '1994-09-23', 'genres': ['Drama'], 'vote_average': 9.3},
    {'title': 'The Godfather', 'release_date': '1972-03-24', 'genres': ['Crime', 'Drama'], 'vote_average': 9.2},
    {'title': 'The Dark Knight', 'release_date': '2008-07-18', 'genres': ['Action', 'Crime', 'Drama'], 'vote_average': 9.0},
    {'title': 'Avatar', 'release_date': '2009-12-18', 'genres': ['Action', 'Adventure', 'Fantasy', 'Sci-Fi'], 'vote_average': 7.8},
    {'title': 'Inception', 'release_date': '2010-07-16', 'genres': ['Action', 'Sci-Fi', 'Thriller'], 'vote_average': 8.8},
    {'title': 'The Matrix', 'release_date': '1999-03-31', 'genres': ['Action', 'Sci-Fi'], 'vote_average': 8.7},
    {'title': 'Forrest Gump', 'release_date': '1994-07-06', 'genres': ['Drama', 'Romance'], 'vote_average': 8.8},
    {'title': 'Pulp Fiction', 'release_date': '1994-10-14', 'genres': ['Crime', 'Drama'], 'vote_average': 8.9},
    {'title': 'The Lord of the Rings: The Return of the King', 'release_date': '2003-12-17', 'genres': ['Adventure', 'Drama', 'Fantasy'], 'vote_average': 9.0},
    {'title': 'Fight Club', 'release_date': '1999-10-15', 'genres': ['Drama', 'Thriller'], 'vote_average': 8.8},
    {'title': 'Interstellar', 'release_date': '2014-11-07', 'genres': ['Adventure', 'Drama', 'Sci-Fi'], 'vote_average': 8.6},
    {'title': 'The Silence of the Lambs', 'release_date': '1991-02-14', 'genres': ['Crime', 'Drama', 'Thriller'], 'vote_average': 8.6},
    {'title': 'Saving Private Ryan', 'release_date': '1998-07-24', 'genres': ['Drama', 'War'], 'vote_average': 8.6},
    {'title': 'The Green Mile', 'release_date': '1999-12-10', 'genres': ['Crime', 'Drama', 'Fantasy'], 'vote_average': 8.6},
    {'title': 'Gladiator', 'release_date': '2000-05-05', 'genres': ['Action', 'Adventure', 'Drama'], 'vote_average': 8.5},
    {'title': 'The Departed', 'release_date': '2006-10-06', 'genres': ['Crime', 'Drama', 'Thriller'], 'vote_average': 8.5},
    {'title': 'Back to the Future', 'release_date': '1985-07-03', 'genres': ['Adventure', 'Comedy', 'Sci-Fi'], 'vote_average': 8.5},
    {'title': 'The Usual Suspects', 'release_date': '1995-08-02', 'genres': ['Crime', 'Drama', 'Mystery'], 'vote_average': 8.5},
    {'title': 'Jurassic Park', 'release_date': '1993-06-11', 'genres': ['Action', 'Adventure', 'Sci-Fi'], 'vote_average': 8.1},
    {'title': 'Terminator 2: Judgment Day', 'release_date': '1991-07-03', 'genres': ['Action', 'Sci-Fi'], 'vote_average': 8.5},
    {'title': 'Titanic', 'release_date': '1997-12-19', 'genres': ['Drama', 'Romance'], 'vote_average': 7.8},
    {'title': 'Avengers: Endgame', 'release_date': '2019-04-26', 'genres': ['Action', 'Adventure', 'Drama'], 'vote_average': 8.4},
    {'title': 'The Lion King', 'release_date': '1994-06-10', 'genres': ['Animation', 'Adventure', 'Drama'], 'vote_average': 8.5},
    {'title': 'E.T. the Extra-Terrestrial', 'release_date': '1982-06-11', 'genres': ['Adventure', 'Family', 'Sci-Fi'], 'vote_average': 7.8},
    {'title': 'Casablanca', 'release_date': '1942-11-26', 'genres': ['Drama', 'Romance', 'War'], 'vote_average': 8.5},
    {'title': 'Citizen Kane', 'release_date': '1941-05-01', 'genres': ['Drama', 'Mystery'], 'vote_average': 8.3},
    {'title': 'Vertigo', 'release_date': '1958-08-21', 'genres': ['Mystery', 'Thriller'], 'vote_average': 8.3},
    {'title': 'Psycho', 'release_date': '1960-12-08', 'genres': ['Horror', 'Mystery', 'Thriller'], 'vote_average': 8.4},
    {'title': 'Jaws', 'release_date': '1975-06-20', 'genres': ['Adventure', 'Drama', 'Thriller'], 'vote_average': 8.0},
    {'title': 'Raiders of the Lost Ark', 'release_date': '1981-06-12', 'genres': ['Action', 'Adventure'], 'vote_average': 8.4},
    {'title': 'The Princess Bride', 'release_date': '1987-10-30', 'genres': ['Adventure', 'Comedy', 'Family', 'Fantasy', 'Romance'], 'vote_average': 8.3},
    {'title': 'Schindler\'s List', 'release_date': '1993-12-15', 'genres': ['Biography', 'Drama', 'History'], 'vote_average': 8.9},
    {'title': 'The Sixth Sense', 'release_date': '1999-08-06', 'genres': ['Drama', 'Mystery', 'Thriller'], 'vote_average': 8.1},
    {'title': 'American Beauty', 'release_date': '1999-10-01', 'genres': ['Drama'], 'vote_average': 8.3},
    {'title': 'Se7en', 'release_date': '1995-09-22', 'genres': ['Crime', 'Drama', 'Mystery', 'Thriller'], 'vote_average': 8.6},
    {'title': 'The Truman Show', 'release_date': '1998-06-05', 'genres': ['Comedy', 'Drama'], 'vote_average': 8.2},
    {'title': 'Good Will Hunting', 'release_date': '1997-12-02', 'genres': ['Drama', 'Romance'], 'vote_average': 8.3},
    {'title': 'The Notebook', 'release_date': '2004-06-25', 'genres': ['Drama', 'Romance'], 'vote_average': 7.8},
    {'title': 'La La Land', 'release_date': '2016-12-09', 'genres': ['Drama', 'Music', 'Romance'], 'vote_average': 8.0},
    {'title': 'Parasite', 'release_date': '2019-10-05', 'genres': ['Drama', 'Thriller'], 'vote_average': 8.6},
    {'title': 'Oppenheimer', 'release_date': '2023-07-21', 'genres': ['Biography', 'Drama', 'History'], 'vote_average': 8.1},
    {'title': 'Dune', 'release_date': '2021-10-22', 'genres': ['Action', 'Adventure', 'Drama', 'Sci-Fi'], 'vote_average': 8.0},
    {'title': 'Barbie', 'release_date': '2023-07-21', 'genres': ['Comedy', 'Fantasy'], 'vote_average': 7.1},
    {'title': 'Oppenheimer', 'release_date': '2023-07-21', 'genres': ['Biography', 'Drama'], 'vote_average': 8.1},
    {'title': 'The Avengers', 'release_date': '2012-05-04', 'genres': ['Action', 'Adventure', 'Sci-Fi'], 'vote_average': 8.0},
    {'title': 'Iron Man', 'release_date': '2008-05-02', 'genres': ['Action', 'Adventure', 'Sci-Fi'], 'vote_average': 7.6},
    {'title': 'Captain America: The Winter Soldier', 'release_date': '2014-04-04', 'genres': ['Action', 'Adventure', 'Sci-Fi'], 'vote_average': 7.7},
    {'title': 'Thor: Ragnarok', 'release_date': '2017-11-03', 'genres': ['Action', 'Adventure', 'Comedy'], 'vote_average': 7.9},
    {'title': 'Black Panther', 'release_date': '2018-02-16', 'genres': ['Action', 'Adventure', 'Sci-Fi'], 'vote_average': 7.3},
]

metadata = pd.DataFrame(movies_data)

# Add required columns
metadata['primary_company'] = 'Movie Studio'
metadata['vote_count'] = np.random.randint(1000, 100000, len(metadata))
metadata['imdb_id'] = [f'tt{i:08d}' for i in range(len(metadata))]
metadata['poster_path'] = None

# Write parquet
metadata.to_parquet(MODEL_DIR / 'movie_metadata.parquet', index=False)

# Create realistic similarity matrix based on genres and ratings
n_movies = len(metadata)
sim_matrix = np.zeros((n_movies, n_movies))

for i in range(n_movies):
    for j in range(n_movies):
        if i == j:
            sim_matrix[i, j] = 1.0
        else:
            # Genre similarity
            genres_i = set(metadata.iloc[i]['genres'])
            genres_j = set(metadata.iloc[j]['genres'])
            if genres_i and genres_j:
                genre_sim = len(genres_i & genres_j) / len(genres_i | genres_j)
            else:
                genre_sim = 0.5
            
            # Rating similarity (inverse of normalized difference)
            rating_diff = abs(metadata.iloc[i]['vote_average'] - metadata.iloc[j]['vote_average'])
            rating_sim = 1.0 - (rating_diff / 10.0)
            
            # Combine similarities
            sim_matrix[i, j] = 0.7 * genre_sim + 0.3 * rating_sim

# Normalize to [0, 1]
sim_matrix = sim_matrix / sim_matrix.max()
np.save(MODEL_DIR / 'similarity_matrix.npy', sim_matrix.astype(np.float32))

# title mapping (lowercase for search)
title_to_idx = {title.lower(): idx for idx, title in enumerate(metadata['title'])}
with open(MODEL_DIR / 'title_to_idx.json', 'w', encoding='utf-8') as f:
    json.dump(title_to_idx, f)

# config
with open(MODEL_DIR / 'config.json', 'w', encoding='utf-8') as f:
    json.dump({'n_movies': len(metadata)}, f)

print(f'Dummy model files written to {MODEL_DIR}')
print(f'Total movies: {len(metadata)}')
