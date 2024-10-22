import os
import face_recognition
import pickle

# Load all images of a person to create encodings
def create_face_data(name, photos_directory):
    encodings = []
    for filename in os.listdir(photos_directory):
        image = face_recognition.load_image_file(os.path.join(photos_directory, filename))
        encoding = face_recognition.face_encodings(image)[0]
        encodings.append(encoding)
    return {name: encodings}


facedata1 = create_face_data('John')