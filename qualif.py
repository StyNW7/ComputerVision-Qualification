# NW25-1, Stanley Nathanael Wijaya

# 0. Import Library

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
import os


# 1. Load Haarcascade


faceCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")


# 2. Training Dataset


train_path = "train"
if not os.path.exists(train_path):
    raise FileNotFoundError("Folder 'train/' tidak ditemukan. Buat folder ini dan isi dengan dataset wajah.")

train_dirs = os.listdir(train_path)
face_list = []
class_list = []
name_list = []

for i, person_name in enumerate(train_dirs):
    person_dir = os.path.join(train_path, person_name)
    if not os.path.isdir(person_dir):
        continue

    name_list.append(person_name)

    for image_path in os.listdir(person_dir):
        path = os.path.join(person_dir, image_path)
        gray = cv.imread(path, 0)
        if gray is None:
            continue

        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        if len(faces) < 1:
            continue

        for (x, y, w, h) in faces:
            face_image = gray[y:y+h, x:x+w]
            face_list.append(face_image)
            class_list.append(i)


# Train face recognizer


if len(face_list) == 0:
    raise ValueError("Tidak ada wajah yang terdeteksi di dataset. Pastikan dataset berisi gambar wajah.")

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(face_list, np.array(class_list))


# 3. Video Capture


vid = cv.VideoCapture(0)
saved_image = None
save_dir = None


while True:
    ret, frame = vid.read()
    if not ret:
        break

    edited_frame = frame.copy()
    gray = cv.cvtColor(edited_frame, cv.COLOR_BGR2GRAY)

    # Image Processing: CLAHE
    clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    cequ_gray = clahe.apply(gray)

    # Face Detection
    faces = faceCascade.detectMultiScale(cequ_gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Blur background, keep face clear
        edited_frame = cv.medianBlur(edited_frame, 35)
        cropped_image = frame[y:y+h, x:x+w]
        edited_frame[y:y+h, x:x+w] = cropped_image

        # Recognize Face
        face_image = gray[y:y+h, x:x+w]
        resIdx, confidence = face_recognizer.predict(face_image)
        text = f"{name_list[resIdx]} {int(confidence)}%"

        cv.rectangle(edited_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(edited_frame, text, (x, y - 10),
                   cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    # Show frame
    cv.imshow("Live Camera", edited_frame)

    if (cv.waitKey(1) & 0xFF == ord(' ')):
        date_now = datetime.now()
        foldername = date_now.strftime("image_%d-%m-%Y-%H%M%S")
        save_dir = os.path.join("images", foldername)
        os.makedirs(save_dir, exist_ok=True)

        filepath = os.path.join(save_dir, "original.jpg")
        cv.imwrite(filepath, frame)
        print(f"[INFO] Saved capture to {filepath}")

        saved_image = frame.copy()
        break


# Release video
vid.release()
cv.destroyAllWindows()


# 4. Post Processing on Captured Image


if saved_image is None:
    raise ValueError("Tidak ada gambar tersimpan. Jalankan ulang dan tekan SPACE untuk capture.")

saved_gray = cv.cvtColor(saved_image, cv.COLOR_BGR2GRAY)

# Blur
blur = cv.blur(saved_image, (10, 10))
blur_rgb = cv.cvtColor(blur, cv.COLOR_BGR2RGB)

# Canny
canny = cv.Canny(saved_gray, 50, 100)

# CLAHE
clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
cequ_gray = clahe.apply(saved_gray)

# Harris Corner (Edge Detection)
harris = cv.cornerHarris(saved_gray, 2, 5, 0.04)
edge = saved_image.copy()
edge[harris > 0.01 * harris.max()] = [0, 0, 255]
edge_rgb = cv.cvtColor(edge, cv.COLOR_BGR2RGB)

# Shape Detection
_, threshold = cv.threshold(cv.medianBlur(cequ_gray, 35), 127, 255, cv.THRESH_BINARY)
contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

shape_img = saved_image.copy()
for contour in contours:
    approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
    M = cv.moments(contour)
    if M["m00"] == 0:
        continue
    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])

    if len(approx) == 3:
        label = "Triangle"
    elif len(approx) == 4:
        label = "Quadrilateral"
    elif len(approx) == 5:
        label = "Pentagon"
    elif len(approx) == 6:
        label = "Hexagon"
    else:
        label = "Circle"

    cv.putText(shape_img, label, (x, y),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

shape_rgb = cv.cvtColor(shape_img, cv.COLOR_BGR2RGB)


# 5. Show Results


labels = ["Blur", "Canny", "Edge Detection", "Shape Detection"]
images = [blur_rgb, canny, edge_rgb, shape_rgb]

plt.figure(figsize=(12, 12))
for i, (lbl, img) in enumerate(zip(labels, images)):
    plt.subplot(2, 2, i + 1)
    plt.imshow(img, cmap="gray" if lbl == "Canny" else None)
    plt.title(lbl)
    plt.axis("off")
plt.tight_layout()

# Save result image to the original photo folder path

result_path = os.path.join(save_dir, "result.jpg")
plt.savefig(result_path)
print(f"[INFO] Saved result to {result_path}")

plt.show()