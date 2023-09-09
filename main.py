import cv2
import face_recognition
import os

# Загрузка базы данных сотрудников
database = {}
for filename in os.listdir('database'):
    if filename.endswith('.jpg'):
        name = os.path.splitext(filename)[0]
        img = face_recognition.load_image_file(os.path.join('database', filename))
        encodings = face_recognition.face_encodings(img)[0]
        database[name] = encodings

# Загрузка изображения
img = cv2.imread('image.jpg')

# Конвертация изображения из BGR в RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Обнаружение лиц на изображении
faces = face_recognition.face_locations(rgb, model='hog')

# Сравнение найденных лиц с лицами из базы данных
for (top, right, bottom, left) in faces:
    encodings = face_recognition.face_encodings(rgb, [(top, right, bottom, left)])[0]
    matches = face_recognition.compare_faces(list(database.values()), encodings)
    name = 'unknown'
    if True in matches:
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}
        for i in matchedIdxs:
            name = list(database.keys())[i]
            counts[name] = counts.get(name, 0) + 1
        name = max(counts, key=counts.get)
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(img, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Отображение изображения с прямоугольниками вокруг лиц и именами сотрудников
cv2.imshow('img', img)
cv2.waitKey()
