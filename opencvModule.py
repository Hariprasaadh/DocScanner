import cv2
import numpy as np


def stackImages(imgArray, scale, labels=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)

    # Get the height and width of the first image
    height, width = imgArray[0][0].shape[:2]

    # Resize images and convert grayscale to BGR if necessary
    if rowsAvailable:
        for x in range(rows):
            for y in range(cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)

        # Create a blank image for horizontal stacking
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows

        # Horizontal stacking of rows
        for x in range(rows):
            hor[x] = np.hstack(imgArray[x])

        # Vertical stacking of rows
        ver = np.vstack(hor)
    else:
        for x in range(rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)

        ver = np.hstack(imgArray)

    # Add labels if provided
    if labels:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)

        for d in range(rows):
            for c in range(cols):
                if d < len(labels) and c < len(labels[d]):
                    # Calculate the maximum label width
                    labelWidth = len(labels[d][c]) * 13 + 27

                    # Ensure the rectangle fits within the image width
                    rectWidth = min(labelWidth, eachImgWidth)

                    # Draw a rectangle for the label
                    cv2.rectangle(ver,
                                  (c * eachImgWidth, eachImgHeight * d),
                                  (c * eachImgWidth + rectWidth, 30 + eachImgHeight * d),
                                  (255, 255, 255),
                                  cv2.FILLED)

                    # Put the text label
                    cv2.putText(ver, labels[d][c],
                                (eachImgWidth * c + 10, eachImgHeight * d + 20),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)

    return ver


def reorder(myPoints):

    myPoints=myPoints.reshape((4,2))
    myPointsNew=np.zeros((4,1,2),dtype=np.int32)

    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]

    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew

    #For points in the Cartesian coordinate system:
          #Top-Left: The point with the smallest sum (smallest x + y).
          #Top-Right: The point with the smallest difference (x - y).
          #Bottom-Right: The point with the largest sum (largest x + y).
          #Bottom-Left: The point with the largest difference (x - y).


def biggestContour(contours):

    biggest = np.array([])
    max_area = 0

    for i in contours:
        area = cv2.contourArea(i)

        if area > 5000:   #Filters contours for optimal processing.

            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)

            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest, max_area


def drawRectangle(img, biggest, thickness):   #biggest Contains largest rectangle vertex coordinates returned by biggestContour() function.

    #biggest = np.array([[[100, 100]],  # Top-left corner
    #                   [[200, 100]],  # Top-right corner
    #                   [[200, 200]],  # Bottom-right corner
    #                    [[100, 200]]])  # Bottom-left corner

    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)

    return img

def nothing(x):
    pass

def initializeTrackbars(intialTracbarVals=0):

    #The initializeTrackbars function creates a GUI with two trackbars for dynamically adjusting threshold values,
    # enabling real-time visualization of their effects on image processing tasks like edge or contour detection.

    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", 200,255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 200, 255, nothing)

def valTrackbars():  #current positions of trackbars in a GUI.

    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    src = Threshold1,Threshold2
    return src






