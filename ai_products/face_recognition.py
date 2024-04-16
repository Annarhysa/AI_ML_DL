# Import libraries
import cv2
from google.colab.patches import cv2_imshow

# Load pre-trained face recognition model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to perform face recognition
def recognize_faces(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Draw bounding boxes around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display the image with bounding boxes
    cv2_imshow(image)

# Test the face recognition function with an example image
image_path = 'IDCard.jpg'  # Replace 'example_image.jpg' with your image path
recognize_faces(image_path)
