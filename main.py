import face_recognition
import os
import cv2
import numpy as np
import math

def face_confidence(face_distance, face_match_threshold=0.6):
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val - 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        if 0 <= linear_val <= 1.0:
            value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2)))
            return str(round(value, 2)) + '%'

class FaceRecognition:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_names = []  # Initialize face_names
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face = face_recognition.load_image_file("faces/" + image)
            encoding = face_recognition.face_encodings(face)
            if len(encoding) > 0:
                encoding = encoding[0].astype(np.float64)  # Convert the data type to float64
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(image)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            sys.exit('Video Source Not Found')

        process_current_frame = True

        try:
            while True:
                ret, frame = video_capture.read()
                if process_current_frame:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = small_frame[:, :, ::-1]
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    face_names = []

                    for face_encoding in face_encodings:
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)

                        if face_distances[best_match_index] < 0.6:
                            name = self.known_face_names[best_match_index]
                            confidence = face_confidence(face_distances[best_match_index])
                            face_names.append(f'{name} ({confidence})')

                    # Update the recognized faces for the current frame
                    self.face_names = face_names

                process_current_frame = not process_current_frame

                frame = cv2.flip(frame, 1)

                for (top, right, bottom, left), name in zip(face_locations, self.face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                cv2.imshow('Face Recognition', frame)

                if cv2.waitKey(1) == ord('q'):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            video_capture.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
