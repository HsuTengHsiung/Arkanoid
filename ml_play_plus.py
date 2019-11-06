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
    comm.ml_ready()

    ball_postition_history=[]
    # 3. Start an endless loop.
    while True:
        # 3.1. Receive the scene information sent from the game process.
        scene_info = comm.get_scene_info()
        platform_center_x=scene_info.platform[0]+20
        ball_postition_history.append(scene_info.ball)
        # 3.2. If the game is over or passed, the game process will reset
        #      the scene and wait for ml process doing resetting job.
        if scene_info.status == GameStatus.GAME_PASS:
            # Do some stuff if needed
            print( "********************************end*****************************************" ,end='\n')
            # 3.2.1. Inform the game process that ml process is ready
            comm.ml_ready()
            continue
        if scene_info.status == GameStatus.GAME_OVER:
            print(scene_info.ball[0],platform_center_x)
            print( 22/0 ,end='\n')
        
        if(len(ball_postition_history) > 1):
            vx=ball_postition_history[-1][0]-ball_postition_history[-2][0]
            vy=ball_postition_history[-1][1]-ball_postition_history[-2][1]

            #i為磚塊編號 j為判斷磚塊垂直邊(0~10)是否在路徑上
            #for i in range(0,len(Brick[0])):
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
                        if( PlatXL <= (BallXR+PlatYT-BallYT+5 )and (BallXR+PlatYT-BallYT+5 )<= PlatXR ):
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
                        if(PlatXL<=(BallXL-(PlatYT-BallYT+5)) and (BallXL-(PlatYT-BallYT+5)<= PlatXR)):
                          #  print("撞到2")
                         #   print(scene_info.bricks[i][0],scene_info.bricks[i][1])
                            flag=1
                            if(len(hit_box)==0):
                                hit_box.append(scene_info.bricks[i][0])
                                hit_box.append(scene_info.bricks[i][1])
                            elif( ( (hit_box[0]-scene_info.ball[0])**2 + (hit_box[1]-scene_info.ball[1])**2 )  >  ( (scene_info.bricks[i][0]-scene_info.ball[0])**2 +  (scene_info.bricks[i][1]-scene_info.ball[1])**2 )):
                                hit_box[0]=scene_info.bricks[i][0]
                                hit_box[1]=scene_info.bricks[i][1]
     #           if(len(hit_box)!=0):
   #                 print("撞到",hit_box[0],hit_box[1])
                    
# =============================================================================
#                 print(X_LR)
#                 print(X_RL)
#                 print(Y_TB)
#                 print(Y_BT)
# =============================================================================
# =============================================================================
#                 if(Y_BT<0 and Y_TB>0):
#                     if(X_LR<0 and X_RL>0):
#                         print("左邊")
#                         print(Brick[0][i][0],Brick[0][i][1])
#                         print("上面")
#                         print()
#                     elif(X_LR>0 and X_RL<0):
#                         print("右邊")
#                         print("上面")
#                 elif(Y_BT>0 and Y_TB<0):
#                     if(X_LR<0 and X_RL>0):
#                         print("左邊")
#                         print("下面")
#                         print(Brick[0][i][0],Brick[0][i][1])
#                         print()
#                     elif(X_LR>0 and X_RL<0):
#                         print("右邊")
#                         print("下面")        
#                                      
# =============================================================================

            if vy > 0 :
                #print("down" ,end='\n')
                down=1
                if vx >0 :
                    final_x =ball_postition_history[-1][0] + (400-ball_postition_history[-1][1])
                elif vx<=0:
                    final_x =ball_postition_history[-1][0] - (400-ball_postition_history[-1][1])      
                if final_x < 0:
                    final_x = 0 - final_x 
                elif final_x>200 :
                    final_x = 400 - final_x -10
                if(len(hit_box)!=0):
                    print(hit_box)
                    if(vx<0):
                     #   print("修正前",final_x)
                        final_x = final_x + (hit_box[0]+25)*2
                    #    print("修正後",final_x)
                    elif(vx>0):
                      #  print("修正前",final_x)                       
                        final_x = final_x - (200-hit_box[0])*2
                       # print("修正後",final_x)
         #       print("final_x=",final_x)
            else:
                #print("up" ,end='\n')
                if(scene_info.ball[1] < 300): #335
                    down=0
                    final_x=100
                elif(vx>0):
                    if(final_x<=165):  
                        final_x += 10
                    if(final_x>=165):
                        final_x=165
                elif(vx<0):
                    if(final_x>=35): 
                        #print(final_x)
                        final_x -= 10
                       # print("修正" ,end="\n")
                        #print(final_x)
                    if(final_x<=35):
                        #15+20
                        final_x=35
                        #print("停止")
# =============================================================================
#             if(final_x>=180):
#                 final_x=180
#             elif(final_x<=20):
#                 final_x=20
# =============================================================================
            print(final_x)
            # 3.3. Put the code here to handle the scene information
            if  (platform_center_x -final_x) >=5:
            # 3.4. Send the instruction for this frame to the game process
                comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
                #print("left" ,end='\n')
            if  (platform_center_x -final_x)<=-5:
                comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)    
                #print("right" ,end='\n')
                
           # print( "ball_x=",ball_postition_history[-1][0] ,end='\n')
            #print( "ball_y=",ball_postition_history[-1][1] ,end='\n')
           # print( "vx=",vx ,end='\n')
           # print( "vy=",vy ,end='\n')
            #print( "finial_platform_center_x =" ,platform_center_x ,end='\n')
            #print( "finial_X =" ,final_x ,end='\n')
            #print( end='\n')
            
                
                
                
                