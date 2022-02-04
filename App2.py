from tkinter import W
import pygame
import sys
import os
import cv2
  
pygame.init()

RES = (400,600)
SCREEN = pygame.display.set_mode(RES)
  
WHITE = (255,255,255)
GREY = (50, 50, 50)
BLACK = (0,0,0)
  
WIDTH = SCREEN.get_width()
HEIGHT = SCREEN.get_height()
  
LARGEFONT = pygame.font.SysFont('Corbel',55)
MEDIUMFONT = pygame.font.SysFont('Corbel',35)
SMALLFONT = pygame.font.SysFont('Corbel',25)

window = True

next_count = 0

quit_button = MEDIUMFONT.render('QUIT' , True , WHITE)
quit_height = 1.1
next_button = MEDIUMFONT.render('NEXT' , True , WHITE)
next_height = 1.25

def take_pic():
    camera = cv2.VideoCapture(0)

    filename = "app_image.jpg"

    while True:
        return_value,image = camera.read()
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        cv2.imshow('image',gray)
        if cv2.waitKey(1)& 0xFF == ord('s'):
            cv2.imwrite(filename,image)
            break

    camera.release()
    cv2.destroyAllWindows()
  
while window:
      
    for ev in pygame.event.get():
          
        if ev.type == pygame.QUIT:
            window = False
            pygame.quit()
              
        #checks if a mouse is clicked
        if ev.type == pygame.MOUSEBUTTONDOWN:
              
            if WIDTH/2-70 <= mouse[0] <= WIDTH/2+70 and HEIGHT/quit_height-20 <= mouse[1] <= HEIGHT/quit_height+20:
                window = False
                pygame.quit()
            elif WIDTH/2-70 <= mouse[0] <= WIDTH/2+70 and HEIGHT/next_height-20 <= mouse[1] <= HEIGHT/next_height+20:
                next_count += 1
                  
    # fills the screen with a color
    SCREEN.fill((WHITE))
      
    mouse = pygame.mouse.get_pos()
      
    if WIDTH/2-70 <= mouse[0] <= WIDTH/2+70 and HEIGHT/quit_height-20 <= mouse[1] <= HEIGHT/quit_height+20:
        pygame.draw.rect(SCREEN,GREY,[WIDTH/2-70,HEIGHT/quit_height-20,140,40])
    else:
        pygame.draw.rect(SCREEN,BLACK,[WIDTH/2-70,HEIGHT/quit_height-20,140,40])

    if WIDTH/2-70 <= mouse[0] <= WIDTH/2+70 and HEIGHT/next_height-20 <= mouse[1] <= HEIGHT/next_height+20:
        pygame.draw.rect(SCREEN,GREY,[WIDTH/2-70,HEIGHT/next_height-20,140,40])
    else:
        pygame.draw.rect(SCREEN,BLACK,[WIDTH/2-70,HEIGHT/next_height-20,140,40])

    def main():
        text_rect = next_button.get_rect(center=(WIDTH/2, HEIGHT/next_height))
        SCREEN.blit(next_button, text_rect)
        text_rect = quit_button.get_rect(center=(WIDTH/2, HEIGHT/quit_height))
        SCREEN.blit(quit_button, text_rect)

        if next_count == 0:
            inital_text_l1 = LARGEFONT.render('CNN', True, BLACK)
            inital_text_l2 = LARGEFONT.render('EMOTION', True, BLACK)
            inital_text_l3 = LARGEFONT.render('PREDICTION', True, BLACK)
            text_rect = inital_text_l1.get_rect(center=(WIDTH/2, HEIGHT/6))
            SCREEN.blit(inital_text_l1, text_rect)
            text_rect = inital_text_l2.get_rect(center=(WIDTH/2, HEIGHT/6+65))
            SCREEN.blit(inital_text_l2, text_rect)
            text_rect = inital_text_l3.get_rect(center=(WIDTH/2, HEIGHT/6+130))
            SCREEN.blit(inital_text_l3, text_rect)
        elif next_count == 1:
            explanation_l1 = SMALLFONT.render('This application uses a CNN', True, BLACK)
            explanation_l2 = SMALLFONT.render('deep learning model to predict', True, BLACK)
            explanation_l3 = SMALLFONT.render('human emotion through image', True, BLACK)
            explanation_l4 = SMALLFONT.render('and audio data', True, BLACK)
            text_rect = explanation_l1.get_rect(center=(WIDTH/2, HEIGHT/6))
            SCREEN.blit(explanation_l1, text_rect)
            text_rect = explanation_l2.get_rect(center=(WIDTH/2, HEIGHT/6+30))
            SCREEN.blit(explanation_l2, text_rect)
            text_rect = explanation_l3.get_rect(center=(WIDTH/2, HEIGHT/6+60))
            SCREEN.blit(explanation_l3, text_rect)
            text_rect = explanation_l4.get_rect(center=(WIDTH/2, HEIGHT/6+90))
            SCREEN.blit(explanation_l4, text_rect)
        elif next_count == 2:
            explanation_l1 = SMALLFONT.render('Please provide the AI with', True, BLACK)
            explanation_l2 = SMALLFONT.render('an image of your face. Choose', True, BLACK)
            explanation_l3 = SMALLFONT.render('a emotion (Happy/Neutral/sad)', True, BLACK)
            explanation_l4 = SMALLFONT.render('then press "s" on keyboard.', True, BLACK)
            text_rect = explanation_l1.get_rect(center=(WIDTH/2, HEIGHT/6))
            SCREEN.blit(explanation_l1, text_rect)
            text_rect = explanation_l2.get_rect(center=(WIDTH/2, HEIGHT/6+30))
            SCREEN.blit(explanation_l2, text_rect)
            text_rect = explanation_l3.get_rect(center=(WIDTH/2, HEIGHT/6+60))
            SCREEN.blit(explanation_l3, text_rect)
            text_rect = explanation_l4.get_rect(center=(WIDTH/2, HEIGHT/6+90))
            SCREEN.blit(explanation_l4, text_rect)
            take_pic()
        elif next_count == 3:
            pass
    
    main()



    pygame.display.update()