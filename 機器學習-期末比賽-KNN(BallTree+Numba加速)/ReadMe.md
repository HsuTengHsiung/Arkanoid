**期末乒乓球對打**
------------------------

遊戲Github:

https://github.com/LanKuDot/MLGame?fbclid=IwAR3n1fCebOJfX9nhzWKnw3wPrVRSX9K1_svT5C_d5Jyz1Zu3InwzQ_LHYGY


環境安裝:

https://hackmd.io/lko7dR0TQ1-3CvSr3hQgTg?view&fbclid=IwAR2k9BzAdpyfj95UyswZwOSPKC0z3LXrM7Ty8oBJfSl_WPxdPFTxORZE9Ls

KNN-BallTree演算法 Github:

https://gist.github.com/dpatschke/f5793db4c1d9cf55b3d16b2fc25c63e3



*****機器學習-桌球PPT為需求設計文件*****
          1.需求
          2.分析
          3.設計
          4.編碼
          5.驗證及修正
![image](https://github.com/HsuTengHsiung/Arkanoid/blob/master/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E6%9C%9F%E6%9C%AB%E6%AF%94%E8%B3%BD-KNN(BallTree%2BNumba%E5%8A%A0%E9%80%9F)/PNG/%E6%95%B4%E9%AB%94%E6%B5%81%E7%A8%8B.PNG)

*****訓練及對打程式:1P/2P資料夾(下面以1P為例)*****
        1P資料夾/:  
          1.訓練用Code: BallTree_1P.py
            執行後會根據路徑去抓取pickle進行訓練，訓練完成後會產生nor.sav(正規化模型)及 BallTree_1P.sav(搜尋模型)
 ![image](https://github.com/HsuTengHsiung/Arkanoid/blob/master/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E6%9C%9F%E6%9C%AB%E6%AF%94%E8%B3%BD-KNN(BallTree%2BNumba%E5%8A%A0%E9%80%9F)/PNG/%E8%A8%93%E7%B7%B4%E6%B5%81%E7%A8%8B.PNG)
         
         2.執行ml_play_BallTree_1P.py
            執行後會根據路徑去抓取nor.sav(正規化模型)及 BallTree_1P.sav(搜尋模型)進行1P的自動對打
            
![image](   https://github.com/HsuTengHsiung/Arkanoid/blob/master/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E6%9C%9F%E6%9C%AB%E6%AF%94%E8%B3%BD-KNN(BallTree%2BNumba%E5%8A%A0%E9%80%9F)/PNG/%E5%B0%8D%E6%89%93%E6%B5%81%E7%A8%8B.PNG)
            
            
