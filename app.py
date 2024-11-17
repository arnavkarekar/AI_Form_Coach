from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

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

def generate_frames():
    """
    Generator function to yield frames from the webcam with pose detection.
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform pose detection
        image_in_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_video.process(image_in_RGB)

        if results.pose_landmarks:
            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(frame,
                                      results.pose_landmarks,
                                      mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(49, 125, 237), thickness=2, circle_radius=2))

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame with multipart data
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """
    Render the main page.
    """
    return render_template('index.html')

@app.route("/test")
def test():
    return render_template('test.html')

@app.route('/video_feed')
def video_feed():
    """
    Route to serve the video feed.
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
