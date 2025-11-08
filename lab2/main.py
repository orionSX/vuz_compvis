import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lr1 = np.array([0, 50, 50])
ur1 = np.array([10, 255, 255])
lr2 = np.array([170, 50, 50])
ur2 = np.array([180, 255, 255])

kernel = np.ones((5, 5), np.uint8)


def spf(image, sp,pp):
   
    noisy = np.copy(image)
    total_pixels = image.size

    
    num_salt = np.ceil(sp * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 1

    
    num_pepper = np.ceil(pp * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 0

    return noisy


image = cv2.imread("D:\Dev\\vuz_compvis\lab3\my_img.jpg")

kart=spf(image,0.3,0.3)

opening = cv2.morphologyEx(kart, cv2.MORPH_OPEN, kernel)

closing = cv2.morphologyEx(kart, cv2.MORPH_CLOSE, kernel)

cv2.imshow("orig", image)

cv2.imshow("salt", kart)
cv2.imshow("Closing", closing)
cv2.imshow("ope", opening)

# while True:
#     ret, frame = cap.read()

#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     kart=spf(frame,0.3,0.3)

#     opening = cv2.morphologyEx(kart, cv2.MORPH_OPEN, kernel)

#     closing = cv2.morphologyEx(kart, cv2.MORPH_CLOSE, kernel)

    

#     # result_frame = frame.copy()

#     # contour=closing
    

    
#     # M = cv2.moments(contour)

#     # if M["m00"] != 0:
#     #     cx = int(M["m10"] / M["m00"])
#     #     cy = int(M["m01"] / M["m00"])
#     #     x, y, w, h = cv2.boundingRect(contour)

#     #     cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 0), 3)

#     #     cv2.circle(result_frame, (cx, cy), 5, (0, 255, 0), -1)

#     # cv2.imshow("Original", frame)
#     # cv2.imshow("HSV", hsv)
#     # cv2.imshow("Red Mask ", red_mask)
#     # cv2.imshow("Red Objects", red_result)
#     # cv2.imshow("Opening ", opening)
#     cv2.imshow("Closing", closing)
    
#     # cv2.imshow("Final Result with Rectangle", result_frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()

if cv2.waitKey(0) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
