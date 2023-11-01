import cv2
import os
import face_recognition
import numpy as np
import math

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range - 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow(abs(linear_val - 0.5) * 2, 0.2)))
        return str(round(value, 2)) + '%'

class FaceRecognition:
    face_location = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face = face_recognition.load_image_file("faces/" + image)
            encoding = face_recognition.face_encodings(face)[0]

            self.known_face_encodings.append(encoding)
            self.known_face_names.append(image)

        print("faces List", self.known_face_names)
    def detect_faces(self, frame):
        if self.process_current_frame:
            self.face_locations = face_recognition.face_locations(frame)
            self.face_encodings = face_recognition.face_encodings(frame, self.face_locations)
            self.face_names = []

            for face_encoding in self.face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding,0.6)
                name = 'Unknown'
                confidence = 'Unknown'

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    confidence = face_confidence(face_distances[best_match_index])

                    self.face_names.append(f'{name} ({confidence})')
                         # find All faces in the current frame
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
                    if len(faces) == 0:
                        return frame

                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
                        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    return frame
                    
        self.process_current_frame = not self.process_current_frame


   
if __name__ == '__main__':
    fr = FaceRecognition()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = fr.detect_faces(frame)
        if frame is not None:
          cv2.imshow('Video Face Detection', frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
