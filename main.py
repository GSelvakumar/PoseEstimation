import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture('PoseVideos/7.mp4')
pTime = 0

while True:
    success, img = cap.read()

    # Screen size
    img = cv2.resize(img, (850, 550))

    # Mediapipe has BGR color but cv supports RGB so convert
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        for id, lm in enumerate(results.pose_landmarks.landmark):
            height, width, channel = img.shape
            print(id, lm)
            # normalized value (0 to 1) is converted to pixels
            cx, cy = int(lm.x * width), int(lm.y * height)
            cv2.circle(img, (cx, cy), 3, (255, 0, 0), cv2.FILLED)

    # to display frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "FPS: "+str(int(fps)), (70, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
