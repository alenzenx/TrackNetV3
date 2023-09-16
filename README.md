# TracknetV3 : beyond TracknetV2

TracknetV3是參考
原始版本的TracknetV2(https://github.com/wolfyeva/TrackNetV2)
及
Resnet版本的TracknetV2(https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2)
加以改進的Tracknet版本，

# 擁有目前Tracknet體系中最先進的精準度，且對於小樣本同樣有最佳的精準度!
# It has the most advanced accuracy in the current Tracknet system, and also has the best accuracy for Few Shot!
## Ours: 90.53%
![image](https://github.com/alenzenx/TracknetV3/blob/main/%E6%9C%80%E6%96%B0%E6%88%90%E6%9E%9C%E8%88%87%E5%8E%9F%E5%A7%8BTracknetV2%20model%E5%B0%8D%E6%AF%94/TracknetV2_encoder%E6%94%B9%E6%88%90%E5%A4%9A%E5%8D%B7%E7%A9%8Dconcat%E4%B8%94%E5%8A%A0%E4%B8%BB%E7%B7%9Achannel%20attention%E5%BE%8C%20concat%E4%B9%8B%E5%89%8D%E4%B9%9F%E5%8A%A0%E5%85%A5channel%20attention/performance.jpg)
## Original: 88.49%
![image](https://github.com/alenzenx/TracknetV3/blob/main/%E6%9C%80%E6%96%B0%E6%88%90%E6%9E%9C%E8%88%87%E5%8E%9F%E5%A7%8BTracknetV2%20model%E5%B0%8D%E6%AF%94/TracknetV2%E5%8E%9F%E5%A7%8B%E8%A8%93%E7%B7%B4/performance.jpg)


如果在 windows native 環境下建議 -> python=3.9.4 、cuda=11.7 、cudnn=8.9.0(cudnn裡的檔案請全部拖進cuda中，不要像 https://medium.com/ching-i/win10-%E5%AE%89%E8%A3%9D-cuda-cudnn-%E6%95%99%E5%AD%B8-c617b3b76deb 這篇文章每項只拖1個檔案進去，請拖該資料夾的全部檔案，不然tensorflow可能會報錯) 

例如: 把 C:\Users\<username>\Downloads\cuda\bin\cudnn64_7.dll 複製到 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin
修正: 把 C:\Users\<username>\Downloads\cuda\bin\ 裡面的所有檔案 複製到 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin

(此配置為 2023/5/15 windows native最頂配)

安裝步驟 : 

獲取小樣本訓練後的最佳權重
https://drive.google.com/file/d/1qh9IiRzZYRY6PbHBGPb6CS0h2AwXPt-d/view?usp=sharing

### 請先檢查有沒有 tutorial-env 的資料夾，有的話請先整個刪除，重新安裝虛擬環境
python 3.9.4安裝時要 Add Python 3.9 to PATH
tensorflow安裝前，請先在 Windows 上啟用長路徑
tensorflow安裝前，要下載並安裝Microsoft Visual C++ Redistributable for Visual Studio 2015、2017 和 2019 。

`python -m venv tutorial-env`                                                                            (安裝虛擬環境)

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`           (安裝pytorch(cuda11.7))

`pip install "tensorflow<2.11"`                                                                          (安裝tensorflow-GPU(大概會安裝tensorflow==2.10.0))

`pip install -r requirements.txt`                                                                        (安裝其他套件)



運行指令備註:
1. zz_Tracknet_badminton_DataConvert.py的檔案是 將imgLabel.py生成的raw_data 轉換成 Tracknetv2-main 所需要的格式。
因為原始資料的標註軟體沒給，所以用imgLabel.py代替，所以需要一個 zz_Tracknet_badminton_DataConvert.py 來轉格式。
2. 請不要用python3，請使用python



運行指令:

### 如何自己做數據集:

### 標註video:(會產生csv)
`python imgLabel.py --label_video_path=你要標註的影片`

### 標註方法(https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2):
![image](https://github.com/alenzenx/TracknetV3/blob/main/%E6%93%8D%E4%BD%9C%E6%89%8B%E5%86%8A%20for%20imgLabel.png)  

全部的影片都標註完成後，請自行分開訓練集與驗證集 並且把 成對的訓練集影片與csv檔 丟到raw_data資料夾 ， 成對的驗證集影片與csv檔 丟到raw_data2資料夾，
並且 TrackNetV2_Dataset資料夾請保持下列形式:

        TrackNetV2_Dataset
                    ├─ train
                    |    
                    |
                    └─ test

除了上述的train與test，TrackNetV2_Dataset底下的其他檔案請都刪除(包括train底下的資料夾與test底下的資料夾)

### 運行 zz_Tracknet_badminton_DataConvert.py : 
注意:運行前，如果是要轉換訓練集，請在 zz_Tracknet_badminton_DataConvert.py 裡更改

`original_raw_data = 'raw_data'`

`target_folder = 'TrackNetV2_Dataset/train'`

如果是要轉換驗證集，請在 zz_Tracknet_badminton_DataConvert.py 裡更改

`original_raw_data = 'raw_data2'`

`target_folder = 'TrackNetV2_Dataset/test'`


### 轉換後 預處理影像:
`python preprocess.py`

### 注意 !!!!! 如果 TrackNetV2_Dataset 裡 已經有 訓練集(train)與驗證集(test) 且2個資料夾裡都有match1、match2...資料夾，即可開始訓練 
### (如果剛拿到專案已經存在的話，代表我已經標註好了，你可以選擇使用我標註的直接訓練，也可以自行標註)

### 訓練:(batchsize請注意:專屬GPU記憶體的大小)
`python train.py --num_frame 3 --epochs 30 --batch_size 8 --learning_rate 0.001 --save_dir exp`

### 預測:
`python predict.py --video_file=test.mp4 --model_file=exp/model_best.pt --save_dir pred_result`

### 預測後使用: 去躁 及 smooth羽球預測的曲線
`python denoise.py --input_csv=pred_result/test_ball.csv`

### smooth羽球預測的曲線後: predict優化後的影片
`python show_trajectory.py --video_file test.mp4 --csv_file pred_result/test_ball.csv --save_dir pred_result`

### smooth羽球預測的曲線後: 預測擊球時刻
`python event_detection.py --input_csv=pred_result/test_ball.csv`