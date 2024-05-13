from ossaudiodev import OSSAudioError
from pathlib import Path
from fastapi import UploadFile
import cv2
import numpy as np
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from mtcnn import MTCNN
import speech_recognition as sr
import librosa
import pickle
import pandas as pd
from .. import schemas
from typing import List
import moviepy.editor as mp
import tempfile
from fer import Video
from fer import FER


BASE_DIR = Path(__file__).resolve(strict=True).parent

# loading trained models
image_model = load_model(f"{BASE_DIR}/image_model.h5")
voice_model = load_model(f"{BASE_DIR}/mk4.h5")
text_model = load_model(f"{BASE_DIR}/text_model.h5")
with open(f"{BASE_DIR}/tokenizer.pkl",'rb') as f:
    tokenizer = pickle.load(f)

###############
#video part
###############

# Create an emotion detector
    emotion_detector = FER(mtcnn=True)
    
def extract_audio(video_file):
    # Load the video file
    video = mp.VideoFileClip(video_file)
    
    # Extract the audio
    audio = video.audio
    
    # Get a temporary file to write the audio data
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        # Write the audio data to the temporary file
        audio.write_audiofile(tmpfile.name, codec='pcm_s16le', ffmpeg_params=['-ac', '1'])  # Mono channel
    
        # Return the path to the temporary file
        return tmpfile.name


def get_emotion_from_video(video_file):
    # Analyze emotions in the video
        video = Video(video_file)
        result = video.analyze(emotion_detector, display=False,output="pandas", save_frames=False, save_video=False, zip_images=False)

        # Drop the 'box' column from DataFrame
        df = result.drop('box', axis=1)

        # Sum up the values in each column
        column_sums = df.sum()

        # Find the total sum of all emotions
        total_sum = column_sums.sum()

        # Calculate the percentages for each emotion
        emotions_percentages = round((column_sums / total_sum) * 100, 2)

        # Sort emotions by percentage in descending order
        emotions_percentages = emotions_percentages.sort_values(ascending=False)

        # Store emotions percentages in a dictionary
        emotion_dict = {}

        for emotion, percentage in emotions_percentages.items():
            emotion_dict[emotion] = percentage

        return emotion_dict


###############
#image part
###############
# Initialize MTCNN for face detection
detector = MTCNN()


# Map class indices to emotion names for image
emotion_mapping = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

# Function to preprocess the input image
def preprocess_image(img):
    img = cv2.resize(img, (48, 48))  # Resize to (48, 48)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = img.reshape(1, 48, 48, 1).astype('float32') / 255.0
    return img

# Function to detect faces in an image
def detect_faces(img):
    faces = detector.detect_faces(img)
    return faces


def get_imotion_from_image(image:UploadFile):
    #loading image
    img = cv2.imdecode(np.frombuffer(image.file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Detect faces in the image
    faces = detect_faces(img)
    
    if len(faces) == 0:
        raise Exception("Ensure your face is clearly visible to the camera for accurate analysis.")
    
    # Iterate over detected faces
    for face in faces:
        # Extract face bounding box coordinates
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)  # Ensure coordinates are not negative
        
        # Extract face ROI
        face_roi = img[y:y+h, x:x+w]
        
        # Preprocess the face ROI
        processed_img = preprocess_image(face_roi)
        
        # Make prediction
        predictions = image_model.predict(processed_img)[0]
        
        # Store the predicted scores in a dictionary with percentages
        image_scores ={}
        for idx, prob in enumerate(predictions):
            emotion = emotion_mapping.get(idx, "Unknown Emotion")
            percentage = round(prob * 100,2)
            image_scores[emotion] = percentage
        
        return image_scores


##############
#audio part
##############

# Define emotions labels for voice
emotions = ['disgust', 'happy', 'sad', 'neutral', 'fear', 'angry']

def analyze_voice(audio):
    try:
        
        # Load and preprocess the audio file
        signal, sample_rate = librosa.load(audio, sr=22050)
        mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
        mfcc = mfcc.T
        mfcc_padded = tf.keras.preprocessing.sequence.pad_sequences([mfcc], padding='post', maxlen=100)

        # Make predictions
        predictions = voice_model.predict(mfcc_padded)
        
        # Get percentages for each emotion
        voice_scores = {emotion: round(predictions[0][i] * 100, 2) for i, emotion in enumerate(emotions)}
        
        return voice_scores
        
    except Exception as e:  # Catch any other unexpected errors
        raise Exception(f"An unexpected error occurred: {e}")
    



###############
#text part
###############

def extract_text_from_audio(audio):


    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(audio) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        return text

    except sr.UnknownValueError:
        raise OSSAudioError("Speech could not be understood. Please try again in a quiet environment, speak clearly, or use a different audio file.")

    except sr.RequestError as e:
        # Handle Google Speech API errors (e.g., network issues)
        raise Exception(f"Error connecting to Google Speech API: {e}")

    except Exception as e:  # Catch any other unexpected errors
        raise Exception(f"An unexpected error occurred: {e}")


# Define maximum sequence length for padding
MAX_SEQUENCE_LENGTH = 30

# Emotion dictionary
emotions_dict = {
    "id_tag": {
        0: "sadness",
        1: "anger",
        2: "love",
        3: "surprise",
        4: "fear",
        5: "joy"
    }
}

def preprocess_text(txt):
        # Convert input text to numerical sequence
        sequences = tokenizer.texts_to_sequences([txt])

        # Pad sequences to ensure consistent input length
        padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

        return padded_sequences 


def analyze_emotions_from_text(txt:str):
        # Preprocess the text input
        processed_text = preprocess_text(txt)

        # Make predictions using the model
        predictions = text_model.predict(processed_text)

        # Map numerical predictions to emotion labels
        text_scores = {}
        for id_, emotion in emotions_dict["id_tag"].items():
            text_scores[emotion] = predictions[0][id_] * 100

        # Return the percentages of each emotion
        return text_scores
##################################
# aggregation for the image part
##################################

def replace_emotions(text_scores):
        # Define a mapping of emotions to be replaced
        replace_map = {
            'love': 'neutral',
            'joy': 'happy',
            'sadness': 'sad',
            'anger': 'angry'
        }
        
        # Create a new dictionary to store the replaced emotions
        replaced_emotions = {}
        
        # Iterate over the items in the original dictionary
        for emotion, percentage in text_scores.items():
            # Check if the emotion needs to be replaced
            if emotion in replace_map:
                # Replace the emotion with the mapped value
                replaced_emotions[replace_map[emotion]] = percentage
            else:
                # If not, keep the original emotion
                replaced_emotions[emotion] = percentage
        
        return replaced_emotions



def add_scores_and_extract_emotion(image_scores, voice_scores,text_scores):
    try:
        # Replace emotions in text_scores
        replaced_emotions = replace_emotions(text_scores=text_scores)
        
        # Combine scores from all dictionaries
        combined_scores = {}
        for emotion in set(image_scores) | set(voice_scores) | set(replaced_emotions):
            # Get the score for each emotion from each dictionary, defaulting to 0 if not present
            score_from_image_scores = image_scores.get(emotion, 0)
            score_from_voice_scores = voice_scores.get(emotion, 0)
            score_from_replaced_emotions = replaced_emotions.get(emotion, 0)
            
            # Sum up the scores
            combined_scores[emotion] = score_from_image_scores + score_from_voice_scores + score_from_replaced_emotions
        
        # Sort combined scores by values in descending order
        sorted_combined_scores = dict(sorted(combined_scores.items(), key=lambda item: item[1], reverse=True))
        
        # Extract the name of the highest score
        highest_score_emotion = next(iter(sorted_combined_scores))
        
        return highest_score_emotion
    except Exception as e:  # Catch any other unexpected errors
        raise Exception(f"An unexpected error occurred: {e}")


#################################
# movies recommendation system
#################################



def recommend_movies_by_mood_with_imdb_links(mood, num_recommendations=5) ->List[schemas.recommended_movie]:
    df = pd.read_csv(f"{BASE_DIR}/movies_df.csv")
    
    # Define the mood categories dictionary
    mood_categories = {
        "sad": ["Drama", "Romance"],
        "happy": ["Animation", "Comedy", "Musical"],
        "surprise": ["Drama", "Comedy", "Romance", "Action", "Adventure"],
        "neutral": ["Documentary", "Drama", "Romance"]
    }
    if mood =='fear':
        mood = 'sad'
    if mood == 'disgust':
        mood = 'angry'

    # Map mood categories to genres
    mood_genres = mood_categories.get(mood, [])
    
    # Filter out rows where genres are not listed or NaN values
    df = df[df['genres'] != '(no genres listed)'].dropna(subset=['genres'])
    
    # Filter movies based on genres matching any of the mood's genres
    filtered_movies = df[df['genres'].apply(lambda x: any(genre.lower() in x.lower() for genre in mood_genres))]
    
    if len(filtered_movies) == 0:
        print("No movies found matching the given mood and genres.")
        return
    
    # Randomly select 5 movies
    recommended_movies = filtered_movies.sample(n=min(num_recommendations, len(filtered_movies)))

    movies = []

    for index, row in recommended_movies.iterrows():
        title = row['Movie']
        language = row['language']
        overview = row['overview']
        imdb_link = f"https://www.imdb.com/title/{row['imdb_id']}"
        movies.append(schemas.recommended_movie(Title=title,language=language,Overview=overview,IMDB_Link=imdb_link))

    return movies



#################################
# music recommendation system
#################################




def recommend_songs_by_mood(mood, num_songs=6)->List[schemas.recommended_song]:
    df3 = pd.read_csv(f"{BASE_DIR}/music_df.csv")


    # Assign cluster names
    cluster_names = {
        0: 'sad',
        1: 'energetic',
        2: 'happy',
        3: 'natural',
        4: 'chill'
        
    }
    if mood == 'surprise':
        mood = 'happy'
    if mood == 'neutral':
        mood = 'natural'
    df3['mood'] = df3['cluster'].map(cluster_names)
    # Filter songs by mood
    mood_songs = df3[df3['mood'] == mood]
    
    if mood_songs.empty:
        # If mood is sad, recommend happy songs
        if mood == "sad" or mood == "fear":
            happy_songs = df3[df3['mood'] == "happy"]
            if not happy_songs.empty:
                mood_songs = happy_songs.sample(min(num_songs, len(happy_songs)))
            else:
                return "No happy songs found"
        
        # If mood is angry, recommend calm songs
        elif mood == "angry" or mood == "disgust":
            calm_songs = df3[df3['mood'] == "chill"]
            if not calm_songs.empty:
                mood_songs = calm_songs.sample(min(num_songs, len(calm_songs)))
            else:
                return "No chill songs found"
        
        # For other moods, return appropriate message
        else:
            print( "No songs found for this mood")
            return
    
    # Sample songs if there are more than the desired number
    if len(mood_songs) > num_songs:
        mood_songs = mood_songs.sample(num_songs)
    elif len(mood_songs) < num_songs:
        mood_songs = mood_songs.sample(len(mood_songs))  # Sample all available songs
    
    # Return list of song titles and artists
    recommended_songs = mood_songs[['track_name']].values.tolist()
    
    songs = []
    for song in recommended_songs:
        songs.append(schemas.recommended_song(Song_name=song[0]))
    return songs