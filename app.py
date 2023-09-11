# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import base64
import cv2

# import the flask dependencies
from flask import Flask, request, jsonify
from flask_cors import CORS

def dectect_and_predict(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))
    
    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    faces = []
    preds = []
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]
        
        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            # add the face to the list
            faces.append(face)
    
    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all* faces at the same time rather than one-by-one predictions in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    
    # return the predicted results
    return preds

def predict_from_data_uri(data_uri):
    
    # Split the data URI to get the base64-encoded part
    data_parts = data_uri.split(',')
    base64_data = data_parts[1]

    # Decode the base64 data
    image_data = base64.b64decode(base64_data)

    # Convert the decoded data to a NumPy array
    image_array = np.frombuffer(image_data, np.uint8)

    # Decode the image array into an OpenCV frame
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Now, 'image' contains the OpenCV frame
    image = imutils.resize(image, width=400)

    # load the face detector model from disk
    prototxtPath = r"./model/face_detector/deploy.prototxt"
    weightsPath = r"./model/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    
    # load the mask detector model from disk
    maskNet = load_model("./model/mask_detector.model")
    
    preds = dectect_and_predict(image, faceNet, maskNet)
    
    # The model is capable for analysing multiple faces at the same time. But we will consider on the first face that get detected for simplification
    if len(preds) == 0:
        return "NO_FACE_DETECTED"
    else:
        (mask, withoutMask) = preds[0]
        if mask > withoutMask:
            return "MASK_DETECTED"
        else:
            return "NO_MASK_DETECTED"

# Create a Flask web app
app = Flask(__name__)

# Configure CORS to allow requests from all origins (this is not recommended for production)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# Define a route that handles POST requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the POST request
        data = request.get_json()
        
        # Access data from the JSON payload
        data_uri = data.get("data_uri")

        # Generating response
        response_data = {
            "status": predict_from_data_uri(data_uri)
        }
        return jsonify(response_data), 200

    except Exception as e:
        # Generating error response
        response_data = {
            "status": "INTERNAL_ERROR"
        }
        return jsonify(response_data), 500

if __name__ == '__main__':
    app.run()