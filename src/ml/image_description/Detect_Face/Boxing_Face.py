import cv2
import matplotlib.pyplot as plt
from PIL import Image

#THIS MODEL WILL NOT DETECT FACES IF MORE THAN 1 FACE IS PRESENT!!!

def cropping_face(img_path):
# load test iamge
    cv_img = cv2.imread(img_path)
    im = Image.open(img_path)
    
    if im.mode != 'RGB':
        im = im.convert('RGB')

    # convert the test image to gray image as opencv face detector expects gray images
    gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # We need to load the required XML classifier.
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    #Detecting the face
    face = face_classifier.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    #cropping the face
    for (x, y, w, h) in face:
        im_cropped = im.crop((x-50, y-40, x + w + 30, y + h +30))

    return im_cropped

if __name__ == "__main__":
    x = cropping_face('src/ml/image_description/Detect_Face/data/test3.jpg')
    plt.figure(figsize=(20,10))
    plt.imshow(x)
    plt.axis('off')
    plt.show()

