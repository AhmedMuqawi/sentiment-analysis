import os
import tempfile
import time
from fastapi import FastAPI,UploadFile
from fastapi.responses import JSONResponse
from .model import model
from . import schemas
app = FastAPI(tags=["Movies & Songs Recommendation"])

@app.post("/image",response_model= schemas.movies_and_songs)
def image_analysis (image : UploadFile,audio: UploadFile):
    try:
        start_time = time.time()  # Start time
        # Create a temporary file to store the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(audio.file.read())
            temp_file_path = temp_file.name
        image_scores = model.get_imotion_from_image(image=image)
        voice_scores = model.analyze_voice(audio=temp_file_path)
        text = model.extract_text_from_audio(audio=temp_file_path)
        text_scores = model.analyze_emotions_from_text(txt=text)
        mood =  model.add_scores_and_extract_emotion(image_scores=image_scores,voice_scores=voice_scores,text_scores=text_scores)
        # #for testing 
        #return mood
        print(mood)
        recommended_movies = model.recommend_movies_by_mood_with_imdb_links(mood=mood)
        recommended_songs = model.recommend_songs_by_mood(mood=mood)
        end_time = time.time()  # End time
        duration = end_time - start_time  # Calculate duration
        print(f"duration: {duration}")
        return schemas.movies_and_songs(recommended_movies=recommended_movies,recommended_songs=recommended_songs)
    except Exception as e:
        # Handle any potential errors during processing
        # print("in except")
        return JSONResponse(status_code=400,content={"Error":f"{e}"})
    finally:
        # Clean up the temporary file (optional)
        if os.path.exists(temp_file_path):
            # print("removing the tempfile")
            os.remove(temp_file_path)





@app.post("/video",response_model= schemas.movies_and_songs)
def image_analysis (video:UploadFile):
    try:
        start_time = time.time()  # Start time
        extracted_audio= ""
        # Create a temporary file to store the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(video.file.read())
            temp_file_path = temp_file.name
        video_scores = model.get_emotion_from_video(video_file=temp_file_path)
        extracted_audio = model.extract_audio(temp_file_path)
        voice_scores = model.analyze_voice(audio=extracted_audio)
        text = model.extract_text_from_audio(audio=extracted_audio)
        text_scores = model.analyze_emotions_from_text(txt=text)
        mood =  model.add_scores_and_extract_emotion(image_scores=video_scores,voice_scores=voice_scores,text_scores=text_scores)
        # #for testing 
        #return mood
        print(mood)
        recommended_movies = model.recommend_movies_by_mood_with_imdb_links(mood=mood)
        recommended_songs = model.recommend_songs_by_mood(mood=mood)
        end_time = time.time()  # End time
        duration = end_time - start_time  # Calculate duration
        print(f"duration: {duration}")
        return schemas.movies_and_songs(recommended_movies=recommended_movies,recommended_songs=recommended_songs)
    except Exception as e:
        # Handle any potential errors during processing
        # print("in except")
        return JSONResponse(status_code=400,content={"Error":f"{e}"})
    finally:
        # Clean up the temporary file (optional)
        if os.path.exists(temp_file_path):
            # print("removing the tempfile")
            os.remove(temp_file_path)
        if os.path.exists(extracted_audio):
            # print("removing the tempfile")
            os.remove(extracted_audio)