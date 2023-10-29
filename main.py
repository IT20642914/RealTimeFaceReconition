import face_recognition
import os,sys
import cv2
import numpy as np
import math

def face_confidence(face_distance,face_match_threshold=0.6):
 range=(1.0-face_match_threshold)
 linear_val=(1.0-face_distance)/(range-2.0)

 if(face_distance>face_match_threshold):
  return str(round(linear_val*100,2))+'%'
 else:
  value=(linear_val+((1.0-linear_val) * math.pow((linear_val-0.5)*2,0.2)))
  return str(round(value,2)) + '%'
 

class FaceRecognition :
 face_location=[]
 face_encodings=[]
 face_names=[]
 known_face_encodings=[]
 known_face_names=[]
 process_current_frame=True

 def __init__(self) :
  self.encode_faces()

def encode_faces(self):
 for image in os.listdir('faces'):
   face_image=face_recognition.load_image_file(f'faces/{image}')
   face_encoding=face_recognition.face_encodings(image)[0]


   self.known_face_encodings.append(face_encoding)
   self.known_face_names.append(image)

   print(self.known_face_names)

if __name__=='__main__':
  fr=FaceRecognition()
  


