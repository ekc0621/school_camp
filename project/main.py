import cv2
import mediapipe as mp
import math

FRAME_DELAY = 50

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_hands = mp.solutions.hands
mp_fingers = mp_hands.HandLandmark


def get_angle(ps, p1, p2):
    angle1 = abs(math.atan((p1.y - ps.y) / (p1.x - ps.x)))
    angle2 = abs(math.atan((p2.y - ps.y) / (p2.x - ps.x)))
    angle = abs(angle1 - angle2) * 180 / math.pi
    return angle

def zoom_at(img, zoom=1, angle=0, coord=None):

    cy, cx = [i/2 for i in img.shape[:-1]] if coord is None else coord[::-1]

    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
    result = cv2.warpAffine(
        img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result

def dist(x1,y1,x2,y2):
    return math.sqrt(math.pow(x1-x2,2)) + math.sqrt(math.pow(y1-y2,2))



def run():
    fing_dist = 1
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print('Ignoring empty camera frame.')
                continue

            image = cv2.flip(image, 1)  # RGB

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #480, 640
            width, height, _ = image.shape
            index_finger_x = 0 
            index_finger_y = 0
            # print(width, height)

            # result processing start
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:

                    index_finger_x = hand_landmarks.landmark[8].x
                    index_finger_y = hand_landmarks.landmark[8].y

                    # dist
                    fing_dist = dist(
                    hand_landmarks.landmark[4].x,
                    hand_landmarks.landmark[4].y,
                    hand_landmarks.landmark[8].x,
                    hand_landmarks.landmark[8].y)
                    fing_dist = (fing_dist*100)-4

                    # angle
                    # index_angle = get_angle(hand_landmarks.landmark[mp_fingers.INDEX_FINGER_MCP],
                    #                         hand_landmarks.landmark[mp_fingers.INDEX_FINGER_DIP],
                    #                         hand_landmarks.landmark[mp_fingers.INDEX_FINGER_TIP])

                    index_finger_tip = hand_landmarks.landmark[mp_fingers.INDEX_FINGER_TIP]

                    # putText
                    cv2.putText(
                        image,
                        text="º",
                        org=(int(index_finger_tip.x * width),
                             int(index_finger_tip.y * height)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 0, 255),
                        thickness=2
                    )

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            fing_dist = fing_dist - 20 / 39
            if (fing_dist < 1): 
                fing_dist = 1
            cv2.imshow('MediaPipe Hands', zoom_at(
                image, fing_dist, coord=(index_finger_x*width , index_finger_y*height)))
            cv2.waitKey(FRAME_DELAY)
        cap.release()


run()