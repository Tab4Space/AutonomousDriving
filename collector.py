import pygame
import os
import csv
import numpy as np
from helper import Agent
from utils import *

CAMERAS = {
    'F_myCamSeg': 5
}

def main():
    subDirPath = makeDirectory(6, CAMERAS)
    a_fd = open(os.path.join('./_data', subDirPath, 'action.csv'), 'w', encoding='utf-8', newline='')
    s_fd = open(os.path.join('./_data', subDirPath, 'state.csv'), 'w', encoding='utf-8', newline='')
    a_wr, s_wr = csv.writer(a_fd), csv.writer(s_fd)

    agent = Agent(True)
    pygame.init()
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    
    tickCnt = 0
    while True:
        # Control agent
        tickCnt += 1
        pygame.event.pump()

        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN and event.button is 10:
                a_fd.close();   s_fd.close()
                os.system('taskkill /f /im AirSimLandscape3.exe'); exit(0)
        joystick.init()

        agent.controller.steering = np.clip(round(joystick.get_axis(0)*2.5, 3), -1.0, 1.0)
        agent.controller.throttle = round(((joystick.get_axis(2)*-1.0)+1.0)/2.0, 3)
        agent.controller.brake = np.clip(round(((joystick.get_axis(3)*-1.0)+1.0)/2.0, 3), 0.0, 1.0)
        agent.client.setCarControls(agent.controller)

        # agent.controller.steering = np.clip(round(joystick.get_axis(0)*2.5, 3), -1., 1.)
        
        if abs(agent.controller.steering) <= 1.0:
            agent.controller.throttle = (0.8-(0.4*abs(agent.controller.steering))) * 0.8
        else:
            agent.controller.throttle = 0.4
        agent.client.setCarControls(agent.controller)

        # Save datas
        camImages = agent.getCamImage(CAMERAS)
        carStates = agent.extractState()
        carActions = agent.getAction()
        
        saveImages(tickCnt, subDirPath, camImages, CAMERAS)
        s_wr.writerow(carStates.tolist())
        a_wr.writerow(carActions.tolist())
        

if __name__=='__main__':
    main()


