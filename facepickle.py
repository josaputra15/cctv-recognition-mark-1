import face_recognition
import pickle
import os

def create_face_data(name, photos_directory):
    encodings = []
    for filename in os.listdir(photos_directory):
        image = face_recognition.load_image_file(os.path.join(photos_directory, filename))
        encoding = face_recognition.face_encodings(image)
        if len(encoding) > 0:
            encodings.append(encoding[0])
    return {name: encodings}

face_data = create_face_data('Ambacong', './img')

with open('face_encodings.pkl', 'wb') as file:
    pickle.dump(face_data, file)
