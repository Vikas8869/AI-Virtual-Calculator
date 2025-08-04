import cv2
import mediapipe as mp
import numpy as np

# Button class
class Button:
    def __init__(self, pos, width, height, value):
        self.pos = pos
        self.width = width
        self.height = height
        self.value = value

    def draw(self, img):
        x, y = self.pos
        cv2.rectangle(img, (x, y), (x + self.width, y + self.height),
                      (70, 70, 70), cv2.FILLED)
        cv2.rectangle(img, (x, y), (x + self.width, y + self.height),
                      (255, 255, 255), 3)
        font_scale = 2 if len(self.value) == 1 else 1.5
        cv2.putText(img, self.value, (x + 20, y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3)

    def checkClick(self, x, y):
        return self.pos[0] < x < self.pos[0] + self.width and \
               self.pos[1] < y < self.pos[1] + self.height

# Button Layout
button_values = [
    ['7', '8', '9', '/'],
    ['4', '5', '6', '*'],
    ['1', '2', '3', '-'],
    ['0', 'DEL', '=', '+']
]
buttons = []
for i in range(4):
    for j in range(4):
        buttons.append(Button((120 * j + 50, 120 * i + 200), 100, 100, button_values[i][j]))

# Mediapipe Hand Setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Text and delay logic
finalText = ""
delayCounter = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Draw Calculator Display
    cv2.rectangle(img, (50, 50), (530, 140), (50, 50, 50), cv2.FILLED)
    cv2.rectangle(img, (50, 50), (530, 140), (255, 255, 255), 3)
    cv2.putText(img, finalText, (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 3)

    # Draw all buttons
    for button in buttons:
        button.draw(img)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for lm in handLms.landmark:
                h, w, _ = img.shape
                lmList.append((int(lm.x * w), int(lm.y * h)))

            if lmList:
                x1, y1 = lmList[4]   # Thumb tip
                x2, y2 = lmList[8]   # Index tip

                # Draw index pointer
                cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # If pinch detected
                if abs(x1 - x2) < 30 and abs(y1 - y2) < 30:
                    if delayCounter == 0:
                        for button in buttons:
                            if button.checkClick(x2, y2):
                                value = button.value
                                if value == "=":
                                    try:
                                        finalText = str(eval(finalText))
                                    except:
                                        finalText = "Error"
                                elif value == "DEL":
                                    finalText = finalText[:-1]
                                else:
                                    finalText += value
                                delayCounter = 1

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Delay counter for single click
    if delayCounter != 0:
        delayCounter += 1
        if delayCounter > 10:
            delayCounter = 0

    cv2.imshow("AI Virtual Calculator", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
