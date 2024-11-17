# from flask import Flask, render_template, Response
# import cv2
# import mediapipe as mp
# import numpy as np

# # Initialize Flask app
# app = Flask(__name__)

# # Initialize Mediapipe Pose
# mp_pose = mp.solutions.pose
# pose_video = mp_pose.Pose(static_image_mode=False,
#                           min_detection_confidence=0.7,
#                           min_tracking_confidence=0.7)
# mp_drawing = mp.solutions.drawing_utils

# # Video capture
# cap = cv2.VideoCapture(0)

# def generate_frames():
#     """
#     Generator function to yield frames from the webcam with pose detection.
#     """
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Perform pose detection
#         image_in_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose_video.process(image_in_RGB)

#         if results.pose_landmarks:
#             # Draw pose landmarks on the frame
#             mp_drawing.draw_landmarks(frame,
#                                       results.pose_landmarks,
#                                       mp_pose.POSE_CONNECTIONS,
#                                       mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
#                                       mp_drawing.DrawingSpec(color=(49, 125, 237), thickness=2, circle_radius=2))

#         # Encode frame as JPEG
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         # Yield frame with multipart data
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     """
#     Render the main page.
#     """
#     return render_template('index.html')

# @app.route("/test")
# def test():
#     return render_template('test.html')

# @app.route('/video_feed')
# def video_feed():
#     """
#     Route to serve the video feed.
#     """
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose_video = mp_pose.Pose(static_image_mode=False,
                          min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Video capture
cap = cv2.VideoCapture(0)

# Initialize counter and stage
counter = 0
stage = 'down'


def calculate_angle(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_degrees = np.degrees(np.arccos(cos_angle))
    return angle_degrees


def get_shoulder_angle(landmarks, side='left'):
    if side == 'left':
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].z]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST].z]
    else:
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].z]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].z]

    vector_shoulder_to_elbow = np.array(elbow) - np.array(shoulder)
    vector_elbow_to_wrist = np.array(wrist) - np.array(elbow)
    angle = calculate_angle(vector_shoulder_to_elbow, vector_elbow_to_wrist)
    return 180 - angle

def get_hip_shoulder_elbow_angle(landmarks, side='left'):
    # Extract coordinates for the left or right side
    if side == 'left':
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, 
               landmarks[mp_pose.PoseLandmark.LEFT_HIP].y, 
               landmarks[mp_pose.PoseLandmark.LEFT_HIP].z]
        
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, 
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y, 
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z]
        
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, 
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y, 
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].z]
    
    elif side == 'right':
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, 
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y, 
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z]
        
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, 
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y, 
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z]
        
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, 
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y, 
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].z]
    
    # Create vectors (using NumPy arrays)
    vector_hip_to_shoulder = np.array(shoulder) - np.array(hip)
    vector_shoulder_to_elbow = np.array(elbow) - np.array(shoulder)
    
    # Calculate the angle between the two vectors
    angle = calculate_angle(vector_hip_to_shoulder, vector_shoulder_to_elbow)
    
    return 180 - angle


def detect_and_draw_selected_landmarks(image_pose, pose):
    global counter, stage

    image_in_RGB = cv2.cvtColor(image_pose, cv2.COLOR_BGR2RGB)
    results = pose.process(image_in_RGB)

    selected_landmarks = {}
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z < landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z:
            relevant_landmarks = [
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.LEFT_WRIST,
                mp_pose.PoseLandmark.LEFT_HIP
            ]

            shoulder_angle = get_shoulder_angle(landmarks, side='left')
            hip_angle = get_hip_shoulder_elbow_angle(landmarks, side='left')  

            if shoulder_angle > 150:
                stage = "down"
            if shoulder_angle < 60 and stage == 'down':
                stage = "up"
                counter += 1
        else:
            relevant_landmarks = [
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_WRIST,
                mp_pose.PoseLandmark.RIGHT_HIP
            ]

            shoulder_angle = get_shoulder_angle(landmarks, side='right')
            hip_angle = get_hip_shoulder_elbow_angle(landmarks, side='right')  

            if shoulder_angle > 150:
                stage = "down"
            if shoulder_angle < 60 and stage == 'down':
                stage = "up"
                counter += 1

        # cv2.putText(image_pose, f'Shoulder Angle: {shoulder_angle:.2f} degrees', (50, 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(image_pose, f'Reps: {counter}', (50, 100),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(image_pose, f'shoulder angle: {shoulder_angle:.2f} degrees', 
                    (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(image_pose, f'hip angle: {hip_angle:.2f} degrees', 
                    (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(image_pose, f'reps: {counter}', 
                    (50, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (255, 255, 255), 2, cv2.LINE_AA)

        # Draw pose landmarks
        # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #                           mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
        #                           mp_drawing.DrawingSpec(color=(49, 125, 237), thickness=2, circle_radius=2))

        for landmark in relevant_landmarks:
            x = int(landmarks[landmark].x * image_pose.shape[1])
            y = int(landmarks[landmark].y * image_pose.shape[0])
            selected_landmarks[landmark.name] = (x, y)

            # Draw a circle on the landmark
            cv2.circle(image_pose, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(image_pose, landmark.name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for i in range(len(relevant_landmarks) - 2):
            start_landmark = relevant_landmarks[i]
            end_landmark = relevant_landmarks[i + 1]

            start_x = int(landmarks[start_landmark].x * image_pose.shape[1])
            start_y = int(landmarks[start_landmark].y * image_pose.shape[0])

            end_x = int(landmarks[end_landmark].x * image_pose.shape[1])
            end_y = int(landmarks[end_landmark].y * image_pose.shape[0])

            # Draw a line between the landmarks
            cv2.line(image_pose, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        start_landmark = relevant_landmarks[0]
        end_landmark = relevant_landmarks[3]

        start_x = int(landmarks[start_landmark].x * image_pose.shape[1])
        start_y = int(landmarks[start_landmark].y * image_pose.shape[0])

        end_x = int(landmarks[end_landmark].x * image_pose.shape[1])
        end_y = int(landmarks[end_landmark].y * image_pose.shape[0])

        # Draw a line between the landmarks
        cv2.line(image_pose, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)

    return image_pose

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform pose detection and draw landmarks
        frame = detect_and_draw_selected_landmarks(frame, pose_video)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/test")
def test():
    return render_template('test.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
