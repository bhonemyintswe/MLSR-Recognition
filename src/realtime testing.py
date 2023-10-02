import gradio as gr
import tensorflow as tf
import numpy as np
import mediapipe as mp
import cv2 
from tensorflow.keras.models import load_model
import traceback


print("Gradio version:", gr.__version__)
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("MediaPipe version:", mp.__version__)
print("OpenCV version:", cv2.__version__)

words = np.array(['မင်္ဂလာပါ','နေကောင်းလား', 'အဆင်ပြေတယ်','တနင်္လာ','အင်္ဂါ','ဗုဒ္ဓဟူး','ကြာသပတေး','သောကြာ','စနေ','တနင်္ဂနွေ'])
model = load_model('/../models/0-9_50Frame_LSTM',compile=False)

mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

empty_frame = np.zeros((480, 480, 3))
predictions_list=[]
threshold = 0.5


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([face, lh, pose, rh])

def classify_video(video):
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        sequence = []
        mean_sequence=[]
                    
        cap = cv2.VideoCapture(video)
        for frame_num in range(1,151):
            ret, frame = cap.read()
            if ret == False or (np.array_equal(empty_frame, frame)):
                frame = last_frame    
            img_resize = cv2.resize(frame, (512, 512))        
            image, results = mediapipe_detection(img_resize, holistic)      
            if results.face_landmarks:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                if len(sequence) == 3:             
                    mean_sequence.append(np.mean(sequence, axis=0))
                    sequence=[]
                                
                                
                if len(mean_sequence) == 50:
                    res = model.predict(np.expand_dims(mean_sequence, axis=0))
                    confidences=res[0]
                    
                    predicted_class = np.argmax(confidences)
                    confidence = float(confidences[predicted_class])
                    if confidence > threshold:
                        sentence=str(words[predicted_class])
                        confidences1 = {words[i]: float(confidences[i]) for i in range(10)}
                    
                
                    
                    
            last_frame = frame
            
    return confidences1


def predict(video_file):
    try:
        confidences1 = classify_video(video_file)
    
    except Exception as e:
        print(traceback.format_exc())
        return "Prediction failed"
    return  confidences1

iface = gr.Interface(
  fn=predict,
  inputs=gr.inputs.Video(),
  outputs=[
    
    gr.outputs.Label(num_top_classes=3),
  ],
)


iface.launch(debug=True)