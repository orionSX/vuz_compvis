import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lr1 = np.array([0, 50, 50])
ur1 = np.array([10, 255, 255])
lr2 = np.array([170, 50, 50])
ur2 = np.array([180, 255, 255])

kernel = np.ones((5, 5), np.uint8)


while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lr1, ur1)
    mask2 = cv2.inRange(hsv, lr2, ur2)

    red_mask = cv2.bitwise_or(mask1, mask2)

    red_result = cv2.bitwise_and(frame, frame, mask=red_mask)

    opening = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    contours, any = cv2.findContours(
        closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result_frame = frame.copy()

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 3000:
            M = cv2.moments(contour)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(contour)

                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 0), 3)

                cv2.circle(result_frame, (cx, cy), 5, (0, 255, 0), -1)

    # cv2.imshow("Original", frame)
    # cv2.imshow("HSV", hsv)
    # cv2.imshow("Red Mask ", red_mask)
    # cv2.imshow("Red Objects", red_result)
    # cv2.imshow("Opening ", opening)
    # cv2.imshow("Closing", closing)
    cv2.imshow("Final Result with Rectangle", result_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
