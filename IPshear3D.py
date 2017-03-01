import cv2
import numpy as np

camera = cv2.VideoCapture('rtsp://192.168.7.163/h264')


ok, image=camera.read()

def nothing(*arg):
        pass

cv2.namedWindow('Features')
#cv2.namedWindow('Test')

#Selecting the method for computing features
#1, 3, 4 fail now
cv2.createTrackbar('aperture', 'Features', 1, 5, nothing)
#Playback error to calculate inliers with RANSAC
cv2.createTrackbar('lambda', 'Features', 532, 1000, nothing)
#Inliers minimum number to indicate that it has recognized an object
cv2.createTrackbar('focal length', 'Features', 160, 200, nothing)
#Trackbar to indicate whether the features are painted or not
cv2.createTrackbar('shearing angle', 'Features', 0, 5, nothing)

c = 0
ok, image=camera.read()
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
prephase = np.zeros(image.shape)

while True:
	
    ok, image=camera.read()
    print(image.shape)
    #image = cv2.resize(image, (0,0), fx=1.0, fy=1.0) 
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #Display
    #cv2.moveWindow('original', 900, 450)
    #cv2.moveWindow('fourier transform', 1375, 450)
    #laplacian = cv2.Laplacian(image,cv2.CV_64F)
    #sobelxy = cv2.Sobel(image,cv2.CV_64F,1,1,ksize=11)

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    mag = np.log(np.abs(fshift))
    mag = 50 * cv2.normalize(mag,1.0,cv2.NORM_L2)

    aperture = cv2.getTrackbarPos('aperture', 'Features')
    lamda = 1/cv2.getTrackbarPos('lambda', 'Features')*1e9
    focal = cv2.getTrackbarPos('focal length', 'Features')
    angle = cv2.getTrackbarPos('shearing angle', 'Features')

    f0 = (np.sin(angle/180*3.1415928)*lamda) / 1000
    fc = ((aperture/1000) * lamda / (focal/1000)) / 1000
    print(f0)
    print(fc)
    print('end')

    cv2.line(mag,(428+np.int(f0+fc),0),(428+np.int(f0+fc),600),(100,0,0),2)
    cv2.line(mag,(428+np.int(f0-fc),0),(428+np.int(f0-fc),600),(100,0,0),2)


    fshift2 = fshift[0:600,428+np.int(f0-fc):428+np.int(f0+fc)]
    fshiftcut = np.zeros(fshift.shape)
    fshiftcut[0:600,428+np.int(f0-fc):428+np.int(f0+fc)] = fshift2
    fshiftcut = np.fft.fftshift(fshiftcut)
    fcut = np.fft.ifft2(fshiftcut)
    fcutR = np.real(fcut)
    fcutI = np.imag(fcut)
    phase = np.arctan(np.divide(fcutI,fcutR))
    imgout = np.uint8((phase + 0.5*np.pi)/np.pi * 255)
    
    diff = np.zeros(image.shape)
    diff = np.abs(phase-prephase)
    totalerror = np.sum(diff)

    prephase = phase

    c = c + 1
    if c < 10001:
        cv2.imwrite('goodfortune2/phase/uphase'+ str(c) + '.bmp',imgout)
        cv2.imwrite('goodfortune2/original/uoriginal'+ str(c) + '.bmp',image)
        print(totalerror,sep= '\n', end= '\r',flush=True)
    else:
        break

    if ok == True:
      cv2.imshow('original',image)
      cv2.imshow('fourier transform',mag)
      cv2.imshow('phase',phase)
      key = cv2.waitKey(1) & 0xFF    
