import cv2
import numpy as np

casc_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # egitilmis model

resimler = [

    'C:/Users/Kerem/Desktop/chris_robert.webp',
    'C:/Users/Kerem/Desktop/kurtlar_vadisi.jpg',
    'C:/Users/Kerem/Desktop/ronaldmessi.jpg'
]

for resim in resimler:

    image = cv2.imread(resim)

    gri_resim = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    tespit_edilen_yuzler = casc_face.detectMultiScale(gri_resim,scaleFactor=1.1,minNeighbors=5)

    for (x, y, w, h) in tespit_edilen_yuzler:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
#                          koordinatlar               renkkodu  kalınlık
    cv2.imshow(f'Yüz Tespiti - {resim}', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
