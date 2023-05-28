from flask import Flask, render_template, request
import pandas as pd
import subprocess
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

app = Flask(__name__)

# Data URL and path
url = "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Music%20Recommendation/spotify_data_urls.csv"
data_path = "./spotify_data_urls.csv"

# Download the file using wget
subprocess.run(["wget", "-O", data_path, url])

# Read the data
data = pd.read_csv(data_path)
all_data = data[['Artist', 'Track', 'Year', 'url', 'Label', 'danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'key', 'loudness', 'mode', 'tempo']]

# Differentiate between numerical and text features
numerical_features = ['Label', 'Year', 'danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'key', 'loudness', 'mode', 'tempo']
text_features = ['Artist', 'Track']

# Define combine_features function
def combine_features(row):
    combined_row = ''
    for feature in text_features:
        combined_row += str(row[feature]) + ' '
    return combined_row[:-1]

# Clean the data by filling null values
if 'genres' in text_features:
    data['genres'] = data['genres'].fillna('')
for feature in text_features:
    data[feature] = data[feature].fillna('')
data["combined_features"] = data.apply(combine_features, axis=1)

# Create feature vectors
cv = CountVectorizer()
count_matrix = cv.fit_transform(data["combined_features"])
text_vectors = count_matrix.toarray()
numerical = data[numerical_features].to_numpy()
numerical = (numerical - numerical.min(axis=0)) / (numerical.max(axis=0) - numerical.min(axis=0))
song_vectors = np.concatenate((text_vectors, numerical), axis=1)

# Calculate similarity scores
def all_similarity(vectors, sim_metric='cosine'):
    if sim_metric == 'cosine':
        return cosine_similarity(vectors)
    else:
        return -pairwise_distances(vectors, metric=sim_metric)

# Define find_index functions
def find_title_from_index(index):
    return data["Track"][index]

def find_artist_from_index(index):
    return data["Artist"][index]

def find_index_from_title(track_name):
    return data.index[data.Track == track_name].values[0]

# Calculate similarity score between two tracks
def similarity_score(track1, track2, vectors, metric='cosine'):
    if track1 is None or track2 is None:
        return None
    similarity_matrix = all_similarity(vectors, metric)
    track_1_index = find_index_from_title(track1)
    track_2_index = find_index_from_title(track2)
    score = similarity_matrix[track_1_index][track_2_index]
    return score

# Song recommendation function :  Your Input here
def SongRec(userinput):
    song_index = find_index_from_title(userinput)
    song_similarity = similarity_matrix[song_index]
    final = sorted(song_similarity, reverse=True)[1:6]
    final_index = [np.where(similarity_matrix == i)[0][1] for i in final]
    final_songs = [find_title_from_index(j) for j in final_index]
    return final_songs

# Routes
@app.route('/')
def user_input_songs():
    df = pd.read_csv(data_path)
    return render_template('./form.html', dataframe=df['Track'])

@app.route('/submit', methods=['POST'])
def process_form():
    df = pd.read_csv(data_path)
    user_input = request.form['user_input']
    recommended_songs = SongRec(user_input)
    return render_template('form.html', songs=recommended_songs, dataframe=df['Track'], userinput= user_input)

# Main
if __name__ == '__main__':
    # Pre-calculate similarity matrix
    similarity_matrix = all_similarity(song_vectors, sim_metric='cosine')
    app.run()
