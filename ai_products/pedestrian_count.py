import cv2

from google.colab.patches import cv2_imshow


# Initialize person counter
person_count = 0

# Load pre-trained person detection classifier (Haar Cascade)
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Read the input image file
image_path = "Screenshot 2024-04-08 143530.png"  # Replace 'image.jpg' with the path to your image file
frame = cv2.imread(image_path)

# Convert the image to grayscale for detection
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect persons in the image
persons = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

# Draw rectangles around detected persons
for (x, y, w, h) in persons:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Update the person count
person_count = len(persons)

# Display the current person count
cv2.putText(frame, f"Persons: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Display the frame
cv2_imshow(frame)