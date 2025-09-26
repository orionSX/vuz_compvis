import cv2

ext_flags = ["jpg", "png", "jpeg"]
read_flags = [cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED]
window_flags = [cv2.WINDOW_AUTOSIZE, cv2.WINDOW_NORMAL, cv2.WINDOW_FREERATIO]

for i in range(len(read_flags)):
    loaded_img = cv2.imread(
        f"/home/orion/dev/vuz/compvis/lab1/peace.{ext_flags[i]}", read_flags[i]
    )
    cv2.namedWindow(f"Image_{i + 1}", window_flags[i])
    cv2.imshow(f"Image_{i + 1}", loaded_img)

cv2.waitKey(0)
cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)
if cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        cv2.imshow("Original", small)
        cv2.imshow("Gray", gray)
        cv2.imshow("HSV", hsv)

        if cv2.waitKey(60) & 0xFF == ord("q"):
            break

    cap.release()
cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)
if cap.isOpened():
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("recorded.avi", fourcc, 20.0, (640, 480))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        out.write(frame)
        cv2.imshow("Recording", frame)

        if cv2.waitKey(60) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()

cv2.destroyAllWindows()


color_img = cv2.imread("/home/orion/dev/vuz/compvis/lab1/peace.png")

hsv_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

cv2.imshow("BGR", color_img)
cv2.imshow("HSV", hsv_img)

cv2.waitKey(0)
cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)
if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        vw, vh = 24, 100
        hw, hh = 100, 24
        cv2.rectangle(
            frame,
            (center_x - vw, center_y - vh),
            (center_x + vw, center_y + vh),
            (0, 0, 255),
            5,
        )
        cv2.rectangle(
            frame,
            (center_x - hw, center_y - hh),
            (center_x + hw, center_y + hh),
            (0, 0, 255),
            5,
        )

        cv2.imshow("t6", frame)

        if cv2.waitKey(60) & 0xFF == ord("q"):
            break

    cap.release()
cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)
if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        center_pixel = frame[center_y, center_x]
        b, g, r = center_pixel[0], center_pixel[1], center_pixel[2]

        distances = [
            abs(r - 255) + abs(g - 0) + abs(b - 0),
            abs(r - 0) + abs(g - 255) + abs(b - 0),
            abs(r - 0) + abs(g - 0) + abs(b - 255),
        ]

        idx = distances.index(min(distances))
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

        cross_color = colors[idx]
        color_names = ["Red", "Green", "Blue"]

        cv2.rectangle(
            frame,
            (center_x - 5, center_y - 40),
            (center_x + 5, center_y + 40),
            cross_color,
            -1,
        )
        cv2.rectangle(
            frame,
            (center_x - 40, center_y - 5),
            (center_x + 40, center_y + 5),
            cross_color,
            -1,
        )

        cv2.imshow("Color Cross", frame)

        if cv2.waitKey(60) & 0xFF == ord("q"):
            break

    cap.release()
cv2.destroyAllWindows()


camera_url = "http://192.168.0.125:8080/video"
cap = cv2.VideoCapture(camera_url)
if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Phone Camera", frame)

        if cv2.waitKey(60) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    cap.release()
