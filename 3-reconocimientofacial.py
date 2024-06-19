import cv2
import os
from tkinter import *
from matplotlib import pyplot
import numpy as np

dataPath = 'data' #Cambia a la ruta donde hayas almacenado Data
numCamara = 0
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)

def cerrarLoginFacial(porcentajevalidacion):
	pantalla.destroy()
	cv2.destroyAllWindows()
	global pantalla_aplicacion
	pantalla_aplicacion = Tk()
	pantalla_aplicacion.geometry("350x250")
	pantalla_aplicacion.title("Applicacion Banca")
	Label(pantalla_aplicacion, text = "Ingreso exitoso con porcentaje: " + str(porcentajevalidacion) ).pack()

def loginFacial():
	face_recognizer = cv2.face.LBPHFaceRecognizer_create()
	# Leyendo el modelo
	face_recognizer.read('modeloLBPHFace.xml')
	cap = cv2.VideoCapture(numCamara,cv2.CAP_DSHOW)

	faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

	while True:
		ret,frame = cap.read()
		if ret == False: break
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		auxFrame = gray.copy()

		faces = faceClassif.detectMultiScale(gray,1.3,5)

		for (x,y,w,h) in faces:
			rostro = auxFrame[y:y+h,x:x+w]
			rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
			result = face_recognizer.predict(rostro)

			cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
			# LBPHFace
			if result[1] >= 70:
				porcentajevalidacion = result[1]
				cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
				# TODO autentica por unos segundos y envia a otra pantalla.
				cerrarLoginFacial(porcentajevalidacion)
				cap.release()
			else:
				cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
			
		cv2.imshow('frame',frame)
		k = cv2.waitKey(1)
		if k == 27:
			break

	cap.release()
	cv2.destroyAllWindows()

def pantalla_principal():
    global pantalla          #Globalizamos la variable para usarla en otras funciones
    pantalla = Tk()
    pantalla.geometry("300x250")  #Asignamos el tamaño de la ventana 
    pantalla.title("Verificación facial")       #Asignamos el titulo de la pantalla
    Label(text = "Login Inteligente", bg = "gray", width = "300", height = "2", font = ("Verdana", 13)).pack() #Asignamos caracteristicas de la ventana
    
#------------------------- Vamos a Crear los Botones ------------------------------------------------------
    
    Label(text = "").pack()  #Creamos el espacio entre el titulo y el primer boton
    Button(text = "Iniciar Sesion Facial", height = "2", width = "30", command = loginFacial).pack()

    pantalla.mainloop()

pantalla_principal()