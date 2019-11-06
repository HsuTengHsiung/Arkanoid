"""The template of the main script of the machine learning process
"""

import games.arkanoid.communication as comm
from games.arkanoid.communication import ( \
    SceneInfo, GameInstruction, GameStatus, PlatformAction
)

def ml_loop():
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
    filename="C:\\Users\\aleal\\OneDrive\\桌面\\MLGame-master\\MLGame-master\\svm.sav"
    model=pickle.load(open(filename, 'rb'))
    
    
          
    filename="C:\\Users\\aleal\\OneDrive\\桌面\\MLGame-master\\MLGame-master\\svm_nor.sav"
    nor_model=pickle.load(open(filename, 'rb'))
    
    comm.ml_ready()

    ball_postition_history=[]
    # 3. Start an endless loop.
    while True:
        
        #***********************************
        
        
        #********************************
        # 3.1. Receive the scene information sent from the game process.
        scene_info = comm.get_scene_info()
        platform_center_x=scene_info.platform[0]
        #***********************************

        
        #********************************
        
        ball_postition_history.append(scene_info.ball)
        
        if(len(ball_postition_history) > 1):
            vx=ball_postition_history[-1][0]-ball_postition_history[-2][0]
            vy=ball_postition_history[-1][1]-ball_postition_history[-2][1]
            hit_box=[]
            for i in range(0,len(scene_info.bricks)):
                BallXL=scene_info.ball[0]   #x1
                BallXR=scene_info.ball[0]+5 #x1+5
                BallYT=scene_info.ball[1]   #y1
                BallYB=scene_info.ball[1]+5 #y1+5
                PlatXL=scene_info.bricks[i][0]       #x2
                PlatXR=scene_info.bricks[i][0]+25    #x2+25
                PlatYT=scene_info.bricks[i][1]       #y2
                PlatYB=scene_info.bricks[i][1]+10    #y2+10
                X_LR=PlatXL-BallXR
                X_RL=PlatXR-BallXL
                Y_TB=PlatYT-BallYB
                Y_BT=PlatYB-BallYT
                if(vy>0 and PlatYT > BallYT):
                    if(vx>0 ):
                        if( PlatXL < (BallXR+PlatYT-BallYT+5 )and (BallXR+PlatYT-BallYT+5 )< PlatXR ):
                            #print("撞到1")
                           # print(scene_info.bricks[i][0],scene_info.bricks[i][1])
                            flag=1
                            if(len(hit_box)==0):
                                hit_box.append(scene_info.bricks[i][0])
                                hit_box.append(scene_info.bricks[i][1])
                            elif( ( (hit_box[0]-scene_info.ball[0])**2 + (hit_box[1]-scene_info.ball[1])**2 )  >  ( (scene_info.bricks[i][0]-scene_info.ball[0])**2 +  (scene_info.bricks[i][1]-scene_info.ball[1])**2 )):
                                hit_box[0]=scene_info.bricks[i][0]
                                hit_box[1]=scene_info.bricks[i][1]
                    elif(vx<0 ):
                        if(PlatXL<(BallXL-(PlatYT-BallYT+5)) and (BallXL-(PlatYT-BallYT+5)< PlatXR)):
                          #  print("撞到2")
                         #   print(scene_info.bricks[i][0],scene_info.bricks[i][1])
                            flag=1
                            if(len(hit_box)==0):
                                hit_box.append(scene_info.bricks[i][0])
                                hit_box.append(scene_info.bricks[i][1])
                            elif( ( (hit_box[0]-scene_info.ball[0])**2 + (hit_box[1]-scene_info.ball[1])**2 )  >  ( (scene_info.bricks[i][0]-scene_info.ball[0])**2 +  (scene_info.bricks[i][1]-scene_info.ball[1])**2 )):
                                hit_box[0]=scene_info.bricks[i][0]
                                hit_box[1]=scene_info.bricks[i][1]
            if(len(hit_box)==0):
                if(vx>0):
                   hit_box.append(200)
                   if(vy>=0):
                       hit_box.append(scene_info.ball[1]+(200-scene_info.ball[0]))
                   if(vy<0):
                       hit_box.append(scene_info.ball[1]-(200-scene_info.ball[0]))
                if(vx<=0):
                   hit_box.append(0)
                   if(vy>=0):
                       hit_box.append(scene_info.ball[1]+(scene_info.ball[0]))
                   if(vy<0):
                       hit_box.append(scene_info.ball[1]-(scene_info.ball[0]))
            elif(len(hit_box)!=0 and vx <0 ):
                 hit_box[0]=hit_box[0]+25
            if(0<hit_box[0]<200):
                print("hit",hit_box[0],hit_box[1])
           # print(vx,vy)
           #*****************************************************************
           
            #********************************************************
            #inp_temp=np.array([scene_info.ball[0],scene_info.ball[1],scene_info.platform[0],scene_info.platform[1],vx,vy])
            inp_temp=np.array([scene_info.ball[0],scene_info.ball[1],scene_info.platform[0],vx,vy,(200-scene_info.ball[0]),(400-scene_info.ball[1]),(scene_info.ball[0]-scene_info.platform[0]),hit_box[0],hit_box[1]])
            #inp_temp=np.array([scene_info.ball[0],scene_info.ball[1],scene_info.platform[0],vx,vy,(200-scene_info.ball[0]),(400-scene_info.ball[1]),(scene_info.ball[0]-scene_info.platform[0])])

            input=inp_temp[np.newaxis, :]
            print(input)
          #  input=nor_model.transform(input)
     #      print(input)
        # 3.2. If the game is over or passed, the game process will reset
        #      the scene and wait for ml process doing resetting job.
        if scene_info.status == GameStatus.GAME_OVER or \
            scene_info.status == GameStatus.GAME_PASS:
            # Do some stuff if needed
            #print( "end" ,end='\n')
            # 3.2.1. Inform the game process that ml process is ready
            comm.ml_ready()
            continue
        
        if(len(ball_postition_history) > 1):
            #print(input)
            move=model.predict(input)
            print(model.predict_proba(input))
            print(move)
        else:
            move = 0
       # print(move)
        if move<-0.05:
            comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
            
        elif move>0.05:
            comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
        
