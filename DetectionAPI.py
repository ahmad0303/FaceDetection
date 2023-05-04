from flask import Flask, request, jsonify
import urllib
import numpy as np
import cv2
import dlib



app = Flask(__name__)


# Load the pre-trained facial landmark detector and face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../content/drive/MyDrive/FaceDetection/shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("../content/drive/MyDrive/FaceDetection/dlib_face_recognition_resnet_model_v1.dat")

@app.route('/match_face', methods=['POST'])
def match_face():
    # Get uri1 and uri2 from the POST request
    uri1 = request.json['uri1']
    uri2 = request.json['uri2']

    # Download the first image to be compared from its URI
    resp1 = urllib.request.urlopen(uri1)
    image1 = np.asarray(bytearray(resp1.read()), dtype="uint8")
    image1 = cv2.imdecode(image1, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    rects1 = detector(gray1, 0)

    # Compute the face embeddings for the first image
    embeddings1 = []
    for rect in rects1:
        shape = predictor(gray1, rect)
        face_embedding = face_recognizer.compute_face_descriptor(image1, shape)
        embeddings1.append(face_embedding)

    # Compare the face embeddings of uri1 with all the images in uri2
    results = []
    for uri in uri2:
        resp = urllib.request.urlopen(uri)
        image2 = np.asarray(bytearray(resp.read()), dtype="uint8")
        image2 = cv2.imdecode(image2, cv2.IMREAD_COLOR)

        # Convert the image to grayscale
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        rects2 = detector(gray2, 0)

        # Compute the face embeddings for the current image in uri2
        embeddings2 = []
        for rect in rects2:
            shape = predictor(gray2, rect)
            face_embedding = face_recognizer.compute_face_descriptor(image2, shape)
            embeddings2.append(face_embedding)

        # Compare the face embeddings of the two images
        for embedding1 in embeddings1:
            for embedding2 in embeddings2:
                distance = np.linalg.norm(np.array(embedding1) - np.array(embedding2))
                if distance < 0.6:
                    results.append({'match': uri, 'distance': distance})

    return jsonify(results)

if __name__ == '__main__':
    app.run()













# from flask import Flask, request, jsonify
# import urllib
# import numpy as np
# import cv2
# import dlib

# app = Flask(__name__)

# # Load the pre-trained facial landmark detector and face recognition model
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
# face_recognizer = dlib.face_recognition_model_v1("./dlib_face_recognition_resnet_model_v1.dat")

# @app.route('/compare_faces', methods=['POST'])
# def compare_faces():
#     # Get the request data
#     data = request.get_json()
#     uri1 = data['uri1']
#     uri2_list = data['uri2_list']
    
#     # Download and decode the images
#     resp1 = urllib.request.urlopen(uri1)
#     image1 = np.asarray(bytearray(resp1.read()), dtype="uint8")
#     image1 = cv2.imdecode(image1, cv2.IMREAD_COLOR)
    
#     images2 = []
#     for uri2 in uri2_list:
#         resp2 = urllib.request.urlopen(uri2)
#         image2 = np.asarray(bytearray(resp2.read()), dtype="uint8")
#         image2 = cv2.imdecode(image2, cv2.IMREAD_COLOR)
#         images2.append(image2)

#     # Convert the images to grayscale
#     gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#     gray2_list = [cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) for image2 in images2]

#     # Detect faces in the grayscale images
#     rects1 = detector(gray1, 0)
#     rects2_list = [detector(gray2, 0) for gray2 in gray2_list]

#     # Compute the face embeddings for each face
#     embeddings1 = []
#     for rect in rects1:
#         shape = predictor(gray1, rect)
#         face_embedding = face_recognizer.compute_face_descriptor(image1, shape)
#         embeddings1.append(face_embedding)

#     embeddings2_list = []
#     for rects2 in rects2_list:
#         embeddings2 = []
#         for rect in rects2:
#             shape = predictor(gray2, rect)
#             face_embedding = face_recognizer.compute_face_descriptor(image2, shape)
#             embeddings2.append(face_embedding)
#         embeddings2_list.append(embeddings2)

#     # Compare the face embeddings
#     result = []
#     for embedding1 in embeddings1:
#         for i, embeddings2 in enumerate(embeddings2_list):
#             for embedding2 in embeddings2:
#                 distance = np.linalg.norm(np.array(embedding1) - np.array(embedding2))
#                 if distance < 0.6:
#                     result.append({"image1": uri1, "image2": uri2_list[i], "match": True})
#                 else:
#                     result.append({"image1": uri1, "image2": uri2_list[i], "match": False})
    
#     return jsonify(result)
