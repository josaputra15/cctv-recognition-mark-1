import cv2
import face_recognition
import pickle
import requests
import subprocess
import os

# 1. Load Face Encodings from File
def load_face_data(filepath='face_encodings.pkl'):
    with open(filepath, 'rb') as file:
        return pickle.load(file)

known_face_data = load_face_data()

# 2. Process Frames and Detect Faces
def process_frame(frame, stream_id):
    # Convert BGR to RGB for face recognition
    rgb_frame = frame[:, :, ::-1]

    # Detect face locations and encodings in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Iterate over detected faces and compare with known faces
    for face_encoding, face_location in zip(face_encodings, face_locations):
        for name, encodings in known_face_data.items():
            matches = face_recognition.compare_faces(encodings, face_encoding)
            if True in matches:
                # Draw a box around the face and label it
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Send an alert via Pub/Sub
                send_alert(name, stream_id)

    return frame

# 3. Send Alert when a face is detected
def send_alert(name, stream_id):
    alert = {
        "event": "FaceDetected",
        "person": name,
        "stream_id": stream_id,
        "timestamp": "2024-10-22T12:00:00Z"
    }

    # Send the alert to a Pub/Sub or webhook endpoint
    response = requests.post('http://pubsub_server_url/alerts', json=alert)
    if response.status_code == 200:
        print(f"Alert sent for {name} on stream {stream_id}")
    else:
        print(f"Failed to send alert: {response.status_code}")

# 4. Setup FFmpeg Streaming Process to RTMP
def start_rtmp_stream(width, height, fps):
    command = [
        'ffmpeg',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}',  # Frame size
        '-r', str(fps),  # Frame rate
        '-i', '-',  # Input from stdin (pipe)
        '-c:v', 'libx264',
        '-f', 'flv',
        'rtmp://ossr_server_url/live/stream'  # RTMP server URL
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)

# 5. Stream Frames to RTMP Server
def stream_frame_to_rtmp(process, frame):
    process.stdin.write(frame.tobytes())

# 6. Handle RTSP Streams, Detect Faces, and Forward Output
def process_rtsp_streams(streams):
    caps = [cv2.VideoCapture(stream) for stream in streams]
    width, height, fps = 640, 480, 30  # Example resolution and frame rate
    rtmp_process = start_rtmp_stream(width, height, fps)

    while True:
        for idx, cap in enumerate(caps):
            ret, frame = cap.read()
            if ret:
                # Process each frame for face recognition
                frame = process_frame(frame, idx)

                # Forward the frame with bounding boxes to RTMP
                stream_frame_to_rtmp(rtmp_process, frame)

                # Optionally display the frame locally (for debugging)
                cv2.imshow(f'Stream {idx}', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    [c.release() for c in caps]
    cv2.destroyAllWindows()

# 7. Main Function to Start Everything
if __name__ == "__main__":
    # Define the RTSP streams (multiple CCTV cameras)
    streams = ['rtsp://camera1_url', 'rtsp://camera2_url']  # Replace with your RTSP streams
    process_rtsp_streams(streams)
