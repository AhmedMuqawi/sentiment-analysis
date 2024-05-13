from pathlib import Path
from fastapi.testclient import TestClient
from .main import app


client = TestClient(app)
BASE_DIR = Path(__file__).resolve(strict=True).parent


def test_image_happy():
    with open(f"{BASE_DIR}/test/happy.jpg", "rb") as image_file, \
            open(f"{BASE_DIR}/test/happy.wav", "rb") as audio_file:
        response = client.post(
            "/image",
            files={"image": ("happy.jpg", image_file), "audio": ("happy.wav", audio_file)}
        )

    assert response.status_code == 200
    assert response.json() == {"mood": "happy"}


def test_image_sad():
    with open(f"{BASE_DIR}/test/sad.jpg", "rb") as image_file, \
            open(f"{BASE_DIR}/test/sad.wav", "rb") as audio_file:
        response = client.post(
            "/image",
            files={"image": ("sad.jpg", image_file), "audio": ("sad.wav", audio_file)}
        )

    assert response.status_code == 200
    assert response.json() == {"mood": "sad"}


def test_image_angry():
    with open(f"{BASE_DIR}/test/angry.jpg", "rb") as image_file, \
            open(f"{BASE_DIR}/test/angry.wav", "rb") as audio_file:
        response = client.post(
            "/image",
            files={"image": ("angry.jpg", image_file), "audio": ("angry.wav", audio_file)}
        )

    assert response.status_code == 200
    assert response.json() == {"mood": "angry"}


def test_image_disgust():
    with open(f"{BASE_DIR}/test/disgust.jpg", "rb") as image_file, \
            open(f"{BASE_DIR}/test/disgust.wav", "rb") as audio_file:
        response = client.post(
            "/image",
            files={"image": ("disgust.jpg", image_file), "audio": ("disgust.wav", audio_file)}
        )

    assert response.status_code == 200
    assert response.json() == {"mood": "disgust"}


def test_image_fear():
    with open(f"{BASE_DIR}/test/fear.jpg", "rb") as image_file, \
            open(f"{BASE_DIR}/test/fear.wav", "rb") as audio_file:
        response = client.post(
            "/image",
            files={"image": ("fear.jpg", image_file), "audio": ("fear.wav", audio_file)}
        )

    assert response.status_code == 200
    assert response.json() == {"mood": "fear"}


def test_image_neutral():
    with open(f"{BASE_DIR}/test/neutral.jpg", "rb") as image_file, \
            open(f"{BASE_DIR}/test/neutral.wav", "rb") as audio_file:
        response = client.post(
            "/image",
            files={"image": ("neutral.jpg", image_file), "audio": ("neutral.wav", audio_file)}
        )

    assert response.status_code == 200
    assert response.json() == {"mood": "neutral"}


def test_image_surprise():
    with open(f"{BASE_DIR}/test/surprise.jpg", "rb") as image_file, \
            open(f"{BASE_DIR}/test/surprise.wav", "rb") as audio_file:
        response = client.post(
            "/image",
            files={"image": ("surprise.jpg", image_file), "audio": ("surprise.wav", audio_file)}
        )

    assert response.status_code == 200
    assert response.json() == {"mood": "surprise"}
