# facial_recognition_app
An app that recognizes faces of anyone using the app and give user granted information on these people. It will also allow you to friend anyone, and share pictures/videos.


# More Information

The concept associated with the design of a facial recognition system is rooted from the basic concept of Linear transformation. And even though it’s very important for us to get a grasp of the technicalities associated, for simplicity, we are going to focus a bit on the maths involved. 

But first to work you through the whole process of how a facial recognition algorithm works. There are three phases to the whole process, face detection which locates a face, face analysis which gives a numerical result, and recognition which compares the result with the database of registered faces. 

Therefore, before any attempt at identification, a database must be built up containing for each user either an initial capture or several. If several captures are made, an average of these is then stored, which allows the system to better adapt to various parameters that may vary at the time of recognition. Once the analysis has been performed, the recognition then performs the comparison between the model obtained and those stored in the database. We then speak of 1 to 1 recognition if we had only one initial catch initially stored in the database, or 1 to N recognition if we had several initials in our database. 

In that regard, facial recognition can be of two types in terms of methodology . 
One of which is 2D recognition, which is considered as the classical method of facial recognition. This method  consists of recognizing the user from a photo of him. This photo can be captured by a camera for security purposes, or simply be already recorded as part of a user assistance session. The recognition is then performed by an algorithm that can be based on different elements, such as the shape of facial elements like the eyes and their spacing, the mouth, and the whole face. 

There is also the 3D recognition method that has replaced that of the 2D. And it’s considered as an improvement of 2D recognition. One way the 3D model is so different from that of a 2D is the fact that it creates a 3D model from several photos taken successively or from a video. This process allows the algorithm to have different points of view of the person to be recognized in order to create the 3D machine learning model. And a machine learning model or algorithm in this case is the data or file that has been trained to recognize certain types of patterns.  And we train a model over a set of data, providing it an algorithm that it can use to reason over and learn from those data.

Okay that must have given us an idea of what facial recognition is,  the different method and the step by step process of how image or face recognition works. Well as I said earlier that we will throw light briefly on the background of the mathematical aspect of image recognition, well this is the right time. 

To begin, facial recognition according to research, is the ability to comprehend, three-dimensional images and shapes. As well as the ability of an algorithm to transform images. 

To help us get a simplified understanding, let us imagine that we have a square image of size 400×400 pixels. This image is represented as a 400×400 matrix. Each element in the matrix will represent the intensity value of the image. Now, if you have an image that needs recognition, machine learning algorithms check the differences between the target image and each of the principal components.

And Principal Component Analysis commonly called PCA for short is an algorithm that reduces the dimensionality of the data while retaining most of the variation in the data set. And in our case we have a 3D image being transformed into a two dimensional image whilst still maintaining its original properties. And it does that by reducing the number of features by choosing the most important ones that still represent the entire dataset. And one of the most important benefits of the process according to findings, is that the dimensionality algorithm can improve the results of the classification task.

PCA is also important in other fields than face recognition like image compression, neuroscience, and computer graphics. 

By facial recognition, we understand privacy, safety, security. As a booming technology, facial recognition has been studied for many years and is expected to be widely used in daily identification systems, communication systems, public security systems, and in law enforcement systems. Many people around the world or even companies have data that they prefer to keep secret. They therefore use facial recognition which is more difficult to crack than a simple pin code, which makes it a very advanced biometric system. Adding biometrics to mobile devices has meant people are actually locking their phones, which secures all their accounts and data, and allows the whole thing to be safely wrapped in encryption instead of creating a strong password you can easily forget. Also, some countries whose technology is quite advanced, use the facial recognition system in some cases to detect criminals in the event of murder, accident or attempted escape (the case of many prisons).
If you have one of the latest iPhone models, you will probably be familiar with facial recognition technology: rather than entering a PIN, there is an option to have the phone scan your face in order to unlock. Although compared with other biometric technologies like iris or fingerprint recognition, facial recognition is less accurate; it is already being widely adopted because it is contactless and non-invasive. Another highly visible use of the technology is on Facebook, which uses facial recognition to suggest which friends to tag in your photos. Some limitations of facial recognition are: 

 Poor Image Quality Limits Facial Recognition's Effectiveness: Image quality affects how well facial-recognition algorithms work. The image quality of scanning video is quite low compared with that of a digital camera.

Small Image Sizes Make Facial Recognition More Difficult: When a face-detection algorithm finds a face in an image or in a still from a video capture, the relative size of that face compared with the enrolled image size affects how well the face will be recognized.

Data Processing and Storage Can Limit Facial Recognition Tech: Even though high-definition video is quite low in resolution when compared with digital camera images, it still occupies significant amounts of disk space. 


# Code - Facial reconition

import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep


def get_encoded_faces():
    """
    looks through the faces folder and encodes all
    the faces

    :return: dict of (name, image encoded)
    """
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded


def unknown_image_encoded(img):
    """
    encode a face given the file name
    """
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding


def classify_face(im):
    """
    will find all of the faces in a given image and label
    them if it knows what they are

    :param im: str of file path
    :return: list of face names
    """
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
    #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    #img = img[:,:,::-1]
 
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)


    # Display the resulting image
    while True:

        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return face_names 


print(classify_face("test.jpg"))


