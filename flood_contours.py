import cv2
import numpy as np
import math
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-in", "--indoor", action='store_true', help="Indoor image")
args = vars(ap.parse_args())

indoor = args["indoor"]

img = cv2.imread(args["image"])
height, width, color = img.shape

if indoor:
    equalized = np.copy(img)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(10, 10))
    for c in range(color):
        equalized[:, :, c] = clahe.apply(equalized[:, :, c])
else:
    equalized = np.copy(img)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
morphed = cv2.morphologyEx(equalized, cv2.MORPH_CLOSE, kernel)

displayed = np.copy(equalized)

seedPoint = None
distancePoint = None
zones = []


def selectSeedPoint(event, x, y, flags, param):
    global seedPoint, distancePoint, zones

    if event == cv2.EVENT_LBUTTONDOWN and not flags & cv2.EVENT_FLAG_CTRLKEY:
        seedPoint = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY:
        distancePoint = (x, y)

    elif event == cv2.EVENT_LBUTTONUP and not flags & cv2.EVENT_FLAG_CTRLKEY:
        if seedPoint is not None:
            zones = sampleFlood(morphed, seedPoint)
            if len(zones) > 0:
                bounds, box, contour, hull = zones[0]

                x, y, w, h = bounds
                rect = cv2.boxPoints(box)
                rect = np.int0(rect)

                cv2.rectangle(displayed, (x, y), (x + w, y + h), (255, 0, 0),
                              2)
                cv2.drawContours(displayed, [rect], 0, (0, 255, 0), 2)
                cv2.drawContours(displayed, [hull], 0, (0, 255, 255), 2)
                cv2.drawContours(displayed, [contour], 0, (255, 255, 0), 2)
                cv2.imshow("image", displayed)

    elif event == cv2.EVENT_LBUTTONUP and flags & cv2.EVENT_FLAG_CTRLKEY:
        if distancePoint is not None:
            if len(zones) > 0:
                bounds, box, contour, hull = zones[0]

                distance = cv2.pointPolygonTest(contour, distancePoint, True)

                cv2.circle(displayed, distancePoint, 5, (255, 255, 255), -1)
                cv2.putText(displayed, str(int(distance)), distancePoint,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))


def sampleFlood(img, seedPoint):
    large_rects = set()
    large_zones = []

    frame = np.copy(img)
    bnw = cv2.addWeighted(frame, 0.9, np.zeros_like(frame), 0.1, 0)
    _, flooded, _, _ = cv2.floodFill(bnw, None, seedPoint, (255, 255, 255),
                                     (10, ) * 3, (10, ) * 3)
    gray = cv2.cvtColor(flooded, cv2.COLOR_BGR2GRAY)
    _, thres = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thres, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2500:
            straight_rect = cv2.boundingRect(contour)

            before_size = len(large_rects)
            large_rects.add(straight_rect)
            after_size = len(large_rects)

            if before_size < after_size:
                rotated_rect = cv2.minAreaRect(contour)
                hull = cv2.convexHull(contour)
                large_zones.append(
                    (straight_rect, rotated_rect, contour, hull))

    return large_zones


cv2.namedWindow("image")
cv2.setMouseCallback("image", selectSeedPoint)

while True:
    cv2.imshow("image", displayed)
    key = cv2.waitKey(1) & 0xFF

    # esc
    if key == 27:
        break
    elif key == ord("r"):
        displayed = np.copy(equalized)

cv2.destroyAllWindows()