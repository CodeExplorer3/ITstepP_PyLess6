import cv2
from PIL import Image

# Пути к файлам
image_face_path = 'face.jpg'
image_mask_path = 'hat.png'
cascade_path = 'haarcascade_frontalface_default.xml'

image_face = cv2.imread(image_face_path)
gray_image = cv2.cvtColor(image_face, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(cascade_path)

face_coordinates = face_cascade.detectMultiScale(
    gray_image,
    scaleFactor=1.1,
    minNeighbors=3,
    minSize=(30, 30)
)

print('Face coordinates:', face_coordinates)

if len(face_coordinates) == 0:
    print("No faces detected. Try adjusting parameters or using another image.")
else:
    face = Image.open(image_face_path)
    hat = Image.open(image_mask_path)
    face = face.convert('RGB')
    hat = hat.convert('RGBA')

    for (x, y, w, h) in face_coordinates:
        resized_hat = hat.resize((w, int(h / 2)))
        face.paste(resized_hat, (x, y - int(h / 2)), resized_hat)

    output_path = 'face_with_hat.jpg'
    face.save(output_path)
    print(f"Saved image with hat to {output_path}")

    face_with_hat = cv2.imread(output_path)
    cv2.imshow('Face with hat', face_with_hat)
    cv2.waitKey()
    cv2.destroyAllWindows()
