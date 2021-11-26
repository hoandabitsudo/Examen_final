# --------------------------------------------------------------------------------------------------------------------
# --------- Examen Final  --------------
# -------------------------------------------------------------------------------------------------------------------
# Juan David Venegas Sanabaria

# Segun lo expresado por correo, esta es la solucion del Examen final, teniendo en cuneta que no
# me funciono el correo al momento que usted envio el mensaje sobre la fecha del mismo.
# -------------------------------------------------------------------------------------------------------------------
# En este archivo se encuentra solucionado los puntos:

# 1) Imagen binaria de los pixel del cesped en blanco

# 2) Encontrar los jugadores y arbitro, marcar cada uno de color rojo e imprimir por consola la cantidad.

# -------------------------------------------------------------------------------------------------------------------

import cv2
import numpy as np

# Abrir imagen
img = cv2.imread('soccer_game.png')

# ------------ Colores -------------------
borde = (0, 0, 255)
color_text = (0, 0, 0)

# Convertir la imagen RGB a HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Definir un intervalo del color azul en HSV
lower_verde = np.array([10, 20, 20])
upper_verde = np.array([100, 255, 255])
lower_rosa1 = np.array([130, 50, 50])
upper_rosa1 = np.array([175, 255, 255])

# Mascara, kernel, erosion, dilatacion, gaussian y Canny
mask = cv2.inRange(hsv, lower_verde, upper_verde)
mask1 = cv2.inRange(hsv, lower_rosa1, upper_rosa1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.ones((8,8), np.uint8)
erosion = cv2.erode(mask, kernel, iterations=6)
dilatation = cv2.dilate(erosion, kernel, iterations=5)
gauss = cv2.GaussianBlur(dilatation, (5,5),1)
Canny = cv2.Canny(gauss, 200, 300)

# Contornos con bordes en rojo
ctns, _ =cv2.findContours(Canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, ctns, -1, borde, 2)

# conteo de jugadores
print("Numero de jugadores :", len(ctns))
text = 'Numero de jugadores :' + str(len(ctns))
cv2.putText(img, text, (50, 400), cv2.FONT_ITALIC, 0.7, color_text, 2)

cv2.imshow('img', img)
#cv2.imshow('gray', gray)
#cv2.imshow('erosion', erosion)
cv2.imshow('Cesped en Blanco', dilatation)
#cv2.imshow('canny', Canny)


cv2.waitKey(0)
#cv2.destroyWindow()