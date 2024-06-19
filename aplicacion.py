import cv2
import numpy as np


# Cargar el clasificador de Haar para la detección de caras
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar la imagen del tatuaje
tattoo_img = cv2.imread('mascara1.png', -1)

# Capturar video desde la cámara
cap = cv2.VideoCapture(0)

cv2.namedWindow('AR Tattoo', cv2.WINDOW_NORMAL)  # Tamaño normalizable
cv2.resizeWindow('AR Tattoo', 1200, 1000) 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar caras en la imagen
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Dibujar un rectángulo alrededor de la cara detectada
        cv2.rectangle(frame, (x - 50, y - 50), (x + w + 50, y + h + 50), (255, 0, 0), 2)

        # Obtener la región de interés (ROI) de la cara para colocar el tatuaje
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Escalar el tatuaje para que coincida con el tamaño de la cara
        tattoo_resized = cv2.resize(tattoo_img, (w + 20, h + 20))

        # Obtener las coordenadas para superponer el tatuaje en la cara
        y1, y2 = min(0, y - 50 ), min(roi_color.shape[0], y + tattoo_resized.shape[0] + 50)
        x1, x2 = min(0, x - 50), min(roi_color.shape[1], x + tattoo_resized.shape[1] + 50)

        # Superponer el tatuaje en la región de interés
        for i in range(y2 - y1):
            for j in range(x2 - x1):
                alpha = tattoo_resized[i, j, 3] / 255.0  # Normalizar el valor de transparencia
                beta = 1.0 - alpha
                roi_color[y1 + i, x1 + j] = (alpha * tattoo_resized[i, j, :3] + beta * roi_color[y1 + i, x1 + j])


    # Mostrar el video con el tatuaje superpuesto
    cv2.imshow('AR Tattoo', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
