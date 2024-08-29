import cv2
from deepface import DeepFace
from PIL import Image

def take_photo(filename='photo.jpg'):
    cap = cv2.VideoCapture(0)  # Open the webcam
    ret, frame = cap.read()  # Capture a single frame
    if ret:
        cv2.imwrite(filename, frame)  # Save the frame to a file
    cap.release()  # Release the webcam
    return filename

# Load the pre-trained face detection model (Haar Cascade)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Capture a photo from the webcam
filename = take_photo()

# Load the captured image
frame = cv2.imread(filename)
if frame is None:
    print("Error: The captured image could not be loaded.")
else:
    # Convert the frame to grayscale for face detection
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_classifier.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Analyze the frame to detect emotion using DeepFace
    dominant_emotion = "Unknown"
    emotion_percentages = {}
    try:
        response = DeepFace.analyze(frame, actions=("emotion",), enforce_detection=False)
        if len(response) > 0:
            emotion_percentages = response[0]["emotion"]  # Get the emotion scores
            dominant_emotion = max(emotion_percentages, key=emotion_percentages.get)  # Get the emotion with the highest percentage

            # Print each emotion's percentage
            print("Emotion Percentages:")
            #for emotion, percentage in emotion_percentages.items():
            #    print(f"{emotion}: {percentage:.2f}%")

            print(f"Dominant Emotion: {dominant_emotion} ({emotion_percentages[dominant_emotion]:.2f}%)")
        else:
            print("No emotion detected.")
    except Exception as e:
        print("Error during emotion analysis:", e)

    # Initialize new_frame with the original frame
    new_frame = frame

    # Draw rectangles and emotion labels around the detected faces
    for (x, y, w, h) in faces:
        cv2.putText(new_frame, text=f"{dominant_emotion}: {emotion_percentages[dominant_emotion]:.2f}%", org=(x, y-10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
        new_frame = cv2.rectangle(new_frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)

    # Convert image to RGB format for displaying with PIL
    frame_rgb = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)

    # Display the processed image
    img_pil.show()
