import cv2
import numpy as np
from keras.models import model_from_json
from deepface import DeepFace


emotion_dict = {0: "sad", 1: "angry", 2: "surprise", 3: "fear", 4: "happy", 5: "disgust", 6: "neutral"}


# load json and create eyes model
eyes_json_file = open('model/eyes_emotion_model.json', 'r')
eyes_loaded_model_json = eyes_json_file.read()
eyes_json_file.close()
eyes_emotion_model = model_from_json(eyes_loaded_model_json)

# load weights into new eyes model
eyes_emotion_model.load_weights("model/emotion_model.h5")
print("Loaded eyes model from disk")

# start the webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    rgb_frame = frame[:, :, ::-1]
    frame = cv2.resize(frame, (1280, 720))

    if not ret:
        break
    
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_mcs_eyepair_big.xml")
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        #face rectangle
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        roi_color = frame[y:y+h, x:x+w]

        result = DeepFace.analyze(rgb_frame, actions = ['emotion'],enforce_detection=False)
        
        #detect eyes available on the camera and Preprocess it
        eyes = eye_cascade.detectMultiScale(roi_gray_frame, scaleFactor=1.3, minNeighbors=5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eye_roi_gray_frame = roi_gray_frame[ey:ey + eh, ex:ex + ew]
            #crop eyes for prediction
            cropped_eyes = np.expand_dims(np.expand_dims(cv2.resize(eye_roi_gray_frame, (48, 48)), -1), 0)
            # predict the emotions with eyes model
            eyes_emotion_prediction = eyes_emotion_model.predict(cropped_eyes)
            eyesMaxindex = int(np.argmax(eyes_emotion_prediction))

            dominant_emote = max(result['emotion'].items(), key=lambda x:x[1])
            
            eye_Emote = emotion_dict[eyesMaxindex]

            if(dominant_emote[1] < 75):
                if(dominant_emote[0] == eye_Emote):
                    emote = dominant_emote[0]
                else:
                    continue
            else : emote = dominant_emote[0]

            font = cv2.FONT_HERSHEY_DUPLEX

            cv2.putText(frame,
                    emote,
                    (50, 50),
                    font, 1,
                    (220, 220, 220),
                    2,
                    cv2.LINE_4)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
