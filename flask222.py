from flask import Flask, Response
import cv2
from webcamdecrpistation import process_frame
#from webcamdecrpistation import video_detection # Import modified continuous capture function
import time

app = Flask(__name__)

# Set the secret key
app.secret_key = 'saishankar'

def generate_frames_web(path_x):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/webapp')
def webapp():
    return Response(generate_frames_web(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/des', methods=['GET'])
def latest_statement():
    statement_generator = capture_and_predict()  # Use continuous capture function
    return Response(generate_statements(statement_generator), mimetype='text/event-stream')

def generate_statements(statement_generator):
    while True:
        statement = next(statement_generator)
        yield f"data: {statement}\n\n"  # Format as Server-Sent Events

if __name__ == "__main__":
    app.run(debug=True)
