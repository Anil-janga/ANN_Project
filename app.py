import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WINDOWSIZEX = 640
WINDOWSIZEY = 480
BOUNDRAYINC = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)  # Define BLACK as (0, 0, 0)
RED = (255, 0, 0)
PREDICT = True
IMAGESAVE = False
MODEL = load_model("bestmodel.keras")
LABELS = {0: "Zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.init()
FONT = pygame.font.Font("freesansbold.ttf", 18)
iswriting = False

pygame.display.set_caption("Digit Board")
number_xcord = []
number_ycord = []
img_cnt = 0

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)
        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDRAYINC, 0), min(WINDOWSIZEX, number_xcord[-1] + BOUNDRAYINC)
            rect_min_Y, rect_max_Y = max(number_ycord[0] - BOUNDRAYINC, 0), min(WINDOWSIZEY, number_ycord[-1] + BOUNDRAYINC)

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.surfarray.pixels3d(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_Y:rect_max_Y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("image.jpg", cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR))
                img_cnt += 1

            if PREDICT:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, (10, 10), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28)) / 255

                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])

                textSurface = FONT.render(label, True, RED, WHITE)
                textRecObj = textSurface.get_rect()
                textRecObj.left, textRecObj.bottom = rect_max_x, rect_min_Y

                DISPLAYSURF.blit(textSurface, textRecObj)

        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)

        pygame.display.update()