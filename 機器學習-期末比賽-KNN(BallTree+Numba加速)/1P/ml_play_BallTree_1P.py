"""The template of the main script of the machine learning process
"""

import games.pingpong.communication as comm
from games.pingpong.communication import (
    SceneInfo, GameInstruction, GameStatus, PlatformAction
)

def ml_loop(side: str):
    """The main loop of the machine learning process

    This loop is run in a separate process, and communicates with the game process.

    Note that the game process won't wait for the ml process to generate the
    GameInstruction. It is possible that the frame of the GameInstruction
    is behind of the current frame in the game process. Try to decrease the fps
    to avoid this situation.
    """

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here.

    # 2. Inform the game process that ml process is ready before start the loop.
    
    
    import pickle
    import numpy as np
    
    filename="C:\\Users\\aleal\\OneDrive\\桌面\\MLGame-master\\MLGame-master\\BallTree_1p.sav"
    model=pickle.load(open(filename, 'rb'))
    
    filename="C:\\Users\\aleal\\OneDrive\\桌面\\MLGame-master\\MLGame-master\\BallTree_nor.sav"
    nor_model=pickle.load(open(filename, 'rb'))
    comm.ml_ready()
    import os
    ball_postition_history=[]
    # 3. Start an endless loop.
    split = 0.67
# =============================================================================
#     model.loadDataset()
# =============================================================================
    while True:
        
        #***********************************
        
        
        #********************************
        # 3.1. Receive the scene information sent from the game process.
        scene_info = comm.get_scene_info()
        platform_center_x=scene_info.platform_1P[0]
        #***********************************
        
        
        #********************************
        
        ball_postition_history.append(scene_info.ball)
        
        if(len(ball_postition_history) > 1):
            vx=ball_postition_history[-1][0]-ball_postition_history[-2][0]
            vy=ball_postition_history[-1][1]-ball_postition_history[-2][1]
           # print(vx,vy)
            #inp_temp=np.array([scene_info.ball[0],scene_info.ball[1],scene_info.platform[0],scene_info.platform[1],vx,vy])
            if vx>=7:
                vx=7
            if vx<=-7:
                vx=-7
            if vy>=7:
                vy=7
            if vy<=-7:
                vy=-7
           # inp_temp=np.array([scene_info.ball[0],scene_info.ball[1],scene_info.platform_1P[0],vx,vy])
            #inp_temp=np.array([scene_info.ball[0],scene_info.ball[1],scene_info.platform_1P[0],vx,vy])
            inp_temp=np.array([scene_info.ball[0],scene_info.ball[1],scene_info.platform_1P[0],(200-scene_info.ball[0]),(scene_info.ball[0]-scene_info.platform_1P[0]),(420-scene_info.ball[1]),vx,vy,scene_info.ball_speed])
            input=inp_temp[np.newaxis, :]
# =============================================================================
#             input=inp_temp
# =============================================================================
            #******************************************************************************************************************************
# =============================================================================
#             print(input)
# =============================================================================
# =============================================================================
#             print(input)
# =============================================================================
            input=nor_model.transform(input)
            input=input[0]
# =============================================================================
#             print(input)
# =============================================================================
            #******************************************************************************************************************************
        # 3.2. If the game is over or passed, the game process will reset
        #      the scene and wait for ml process doing resetting job.
        if scene_info.status == GameStatus.GAME_1P_WIN or \
           scene_info.status == GameStatus.GAME_2P_WIN:
            # Do some updating or resetting stuff
            print(scene_info.ball_speed)
            os.system('pause')
            # 3.2.1 Inform the game process that
            #       the ml process is ready for the next round
            comm.ml_ready()
            continue
            
        
        if(len(ball_postition_history) > 1):
# =============================================================================
#             move=model.predict(input)
# =============================================================================
# =============================================================================
#             print(input)
# =============================================================================
            print(input)
            move=model.query(input[np.newaxis, :], 1)
            print(move)
# =============================================================================
#             print("ball" ,scene_info.ball)
#             print("plat" ,scene_info.platform_1P[0])
# =============================================================================
        else:
            move = 99
# =============================================================================
#         if move==7 or move==8 or move==11:
#             comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
#             
#         elif move==9 or move==0  or move==2 or move==14 or move==10 or move==4 or move==12:
#             comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
# =============================================================================
        if move==-1:
            comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
            
        elif move==1:
            comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)        
