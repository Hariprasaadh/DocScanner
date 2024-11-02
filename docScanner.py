import cv2
import numpy as np
import opencvModule as ocm

pathImage = "img1.jpg"
imgW = 640
imgH = 480

ocm.initializeTrackbars()

while True:
    img = cv2.imread(pathImage)
    if img is None:
        print("Image not found. Please check the path.")
        break

    img = cv2.resize(img, (imgW, imgH))  # RESIZE IMAGE
    imgBlank = np.zeros((imgH, imgW, 3), np.uint8)  # CREATE A BLANK IMAGE
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    thres = ocm.valTrackbars()  # GET TRACK BAR VALUES FOR THRESHOLDS
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])
    kernel = np.ones((5, 5))

    # Dilation and Erosion
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION

    imgContours = img.copy()
    imgBigContour = img.copy()
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

    biggest, maxArea = ocm.biggestContour(contours)  # FIND THE BIGGEST CONTOUR
    if biggest.size != 0:
        biggest = ocm.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
        imgBigContour = ocm.drawRectangle(imgBigContour, biggest, 2)

        # Define source and destination points for perspective transformation
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [imgW, 0], [0, imgH], [imgW, imgH]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        imgWarpColored = cv2.warpPerspective(img, matrix, (imgW, imgH))

        # REMOVE 20 PIXELS FORM EACH SIDE
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored, (imgW, imgH))

        # APPLY ADAPTIVE THRESHOLD
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)  # Invert binary image
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)  # Remove noise

        # 1. Gaussian Blur
        imgWarped = cv2.GaussianBlur(imgWarpColored, (5, 5), 0)

        # 2. Sharpening
        kernel_sharpen = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
        imgWarped = cv2.filter2D(imgWarped, -1, kernel_sharpen)

        # 3. CLAHE
        gray = cv2.cvtColor(imgWarped, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        imgWarped = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

        # 4. Enhance Colors (if needed)
        hsv = cv2.cvtColor(imgWarped, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.2  # Increase saturation
        imgWarpedColored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 5. Display the enhanced warped image
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])
    else:  # When no page is found
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    labels = [["Original", "Gray", "Threshold", "Contours"],
              ["Biggest Contour", "Warp Perspective", "Warp Gray", "Adaptive Threshold"]]

    stackedImage = ocm.stackImages(imageArray, 0.75, labels)
    cv2.imshow("Result", stackedImage)

    # Check if 'S' key is pressed to save the image
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("imgWarpColored.jpg", imgWarpColored)
        print("Image saved as imgWarpColored.jpg")

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
