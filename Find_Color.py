import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(3, 300)
cap.set(4, 300)


background = [0, 0, 77, 255, 195, 255]

myColors = {
    "Red": [0, 91, 157, 183, 244, 291],
    "Yello": [21, 71, 100, 70, 255, 172],
    "Green": [42, 89, 74, 106, 238, 179],
}
# myColorValues = [[51, 153, 255], [255, 0, 255], [0, 255, 0], [255, 0, 0]]  ## BGR


def findColor(result, color):
    imgHSV = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    lower = np.array(color[0:3])
    upper = np.array(color[3:6])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(result, result, mask=mask)
    return imgResult


def get_key_from_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None


def getContours(imgResult, img):
    imgHSV1 = cv2.cvtColor(imgResult, cv2.COLOR_BGR2HSV)
    lower1 = np.array(background[0:3])
    upper1 = np.array(background[3:6])
    mask1 = cv2.inRange(imgHSV1, lower1, upper1)
    imgResult = cv2.bitwise_and(imgResult, imgResult, mask=mask1)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contours1, hierarchy1 = cv2.findContours(
        img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    for cnt1 in contours1:
        area1 = cv2.contourArea(cnt1)
        if area1 > 200:
            peri1 = cv2.arcLength(cnt1, True)
            approx1 = cv2.approxPolyDP(cnt1, 0.02 * peri1, True)
            x1, y1, w1, h1 = cv2.boundingRect(approx1)

            for color in myColors.values():
                imgResult = findColor(img, color)
                imgResult = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
                contours, hierarchy = cv2.findContours(
                    imgResult, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 100:
                        peri = cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                        x, y, w, h = cv2.boundingRect(approx)
                        objCor = len(approx)

                        if objCor == 3:
                            objectType = "Tri"
                        elif objCor == 4:
                            aspRatio = w / float(h)
                            if aspRatio > 0.98 and aspRatio < 1.03:
                                objectType = "Square"
                            else:
                                objectType = "Rectangle"
                        elif objCor > 4:
                            objectType = "Circles"
                        else:
                            objectType = "None"

                        cv2.rectangle(
                            img,
                            (x, y),
                            (x + w, y + h),
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            img,
                            get_key_from_value(myColors, color),
                            (x + (w // 2), y + (h // 2) - 10),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                        cv2.putText(
                            img,
                            objectType,
                            (x + (w // 2), y + (h // 2) + 20),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                        box = cv2.minAreaRect(cnt1)
                        box = cv2.boxPoints(box)
                        box = np.array(box, dtype="int")
                        for a, b in box:
                            cv2.putText(
                                img,
                                str(a) + "+" + str(b),
                                (a, b),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.7,
                                (255, 255, 255),
                                2,
                            )
    return img


while True:
    success, img = cap.read()
    if img is None:
        break
    imgResult = img.copy()
    img = getContours(imgResult, img)
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
