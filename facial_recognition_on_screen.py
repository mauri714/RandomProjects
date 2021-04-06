import numpy as np
import cv2

from PIL import ImageGrab
import time


def captured_screen(resolution_requested):

    # inputted list is in the form of a list
    # the first element of that list = x, second = y coordinates
    # 1920x1080p capture will be [1920, 1080]
    x_dimension = resolution_requested[0]
    y_dimentsion = resolution_requested[1]

    # screen = numpyArray that is generated from the
    # imageGrab of the requested resolution/section of the screen
    # bbox is the defined coordinates for the screen
    # "0, 0" in bbox is where to define where to start counting from (pixels on screen)
    screen = np.array(ImageGrab.grab(bbox=(0, 0, x_dimension, y_dimentsion)))

    #returns the screen array (rgb info)
    return screen

def screen_faceEyefinder(screen_size):
    last_time = time.time()
    # while TRUE so that
    # it runs the entire time
    while(True):
        # defined resolution
        screen = captured_screen(screen_size)

        # convert to grayscale for easier and quicker processing by cv2
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        # defining cascades/templates/target --- "\file\location\"
        face_cascade = cv2.CascadeClassifier('C:\Python27\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('C:\Python27\Lib\site-packages\cv2\data\haarcascade_eye.xml')

        # list of lists [[],[]] of face(s) that are detected are returned as Rect[x,y,w,h]
        # gray = grayscale screen defined earlier @ line 35
        # 1.3 = scalefactor (responsible for downscaling [not upscaling b/z that introduces artifacts] potential faces captured
        # to faces in the cascade, and or the template applied. So large faces captured on screen
        # will be downscaled and compared to the smaller faces in the cascade file, making it much
        # easier for the function to determine if it is a face or not. A scale factor of 1.05 means
        # reduce the size by 5%, you increase the chance of a matching size with the model for detection.)
        # 5 = miniNeighbors (higher value = less detections, but greater quality of detection)
        faces = face_cascade.detectMultiScale(gray, 1.05, 5)

        # loop to create square(s) to surround face(s)
        for (x, y, w, h) in faces:
            # image to apply it to = screen
            # start point = (x,y) --- top left of square
            # end point = (x+w, y+h) --- bottom right of square
            # cv2 creates the square automatically based on these 2 points
            # thickness of square = 2
            cv2.rectangle(screen, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # region of interest for eye detection --- numpy slicing
            # slices the gray array
            # row start with y till y+h
            # column start with x till x+h
            roi_gray = gray[y:y + h, x:x + w] # returns the cropped face from the image
            roi_color = screen[y:y + h, x:x + w] # returns details of the image that you will receive after getting co-ordinates of the image

            # detect eyes from roi
            eyes = eye_cascade.detectMultiScale(roi_gray)

            # loop to create square(s) to surround eye(s)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


        print("FPS: {:.2f}".format(1.0 / (time.time() - last_time)))
        print("Loop Took: {:.2f}".format(time.time() - last_time))
        last_time = time.time()
        cv2.imshow('captured screen', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))


        # if q key pressed, quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

# Enter requested resolution as a list. X as index 0, Y as index 1.
screen_faceEyefinder([1265,862])