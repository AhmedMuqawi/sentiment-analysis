from pydantic import BaseModel
from typing import List



#pydantic model for movies 
class recommended_movie (BaseModel):
    Title: str
    language: str
    Overview: str
    IMDB_Link : str


#pydantic model for songs
class recommended_song (BaseModel):
    Song_name: str


#pydantic model for recommended movies and songs
class movies_and_songs (BaseModel):
    recommended_movies : List[recommended_movie]
    recommended_songs : List[recommended_song]