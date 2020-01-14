**期末乒乓球對打**
------------------------     
     機器學習-桌球PPT為需求設計文件
          1.需求
          2.分析
          3.設計
          4.編碼
          5.驗證及修正
          
      訓練及對打程式:1P/2P資料夾
        1P資料夾/:  
          1.訓練用Code: BallTree_1P.py
            執行後會根據路徑去抓取pickle進行訓練，訓練完成後會產生nor.sav(正規化模型)及 BallTree_1P.sav(搜尋模型)
          ![image](https://github.com/HsuTengHsiung/Arkanoid/blob/master/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E6%9C%9F%E6%9C%AB%E6%AF%94%E8%B3%BD-KNN(BallTree%2BNumba%E5%8A%A0%E9%80%9F)/Capture.PNG)

          2.執行ml_play_BallTree_1P.py
            執行後會根據路徑去抓取nor.sav(正規化模型)及 BallTree_1P.sav(搜尋模型)進行1P的自動對打
