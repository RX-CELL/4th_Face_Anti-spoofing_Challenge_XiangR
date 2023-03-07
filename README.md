# 4th Chalearn Face Anti-spoofing Workshop and Challenge@CVPR2023
# Team:&emsp;XiangR &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Date:&emsp;2023.3.7
<br />  
<br />  

# Prerequisites
## 1.1 How to download Surveillance High-Fidelity Mask (SuHiFiMask) dataset?

1. [Surveillance Face Anti-spoofing](https://arxiv.org/abs/2301.00975)
2. Download, read the [Contest Rules](https://codalab.lisn.upsaclay.fr/competitions/10080), and sign the agreement
3. Send the your signed agreements  to Jun Wan, jun.wan@ia.ac.cn 
4. Download and extract the data and organize it into the following 1.2.2 structure
## 1.2 Index tree of data in the project
```
4th_Face_Anti-spoofing_Challenge
├── train (Training sets, development sets, and test sets, along with the corresponding path files)
│── dev
├── test
│── train_label.txt
│── dev.txt
│── test.txt
│── weights (The weight file for the model)
|   ├── convnext_xlarge_22k_224.pth
|   ├── best_model.pth (This file is generated after running train.py)
|   ├── latest_model.pth (This file is generated after running train.py)
│── model.py
│── train.py
│── predict_batch.py
│── utils.py
│── README.md
│── phase1.txt (This file is generated after running predict_batch.py)
│── pre_test.txt (This file is generated after running predict_batch.py)
│── phase2.txt (This file is generated after running predict_batch.py)
```

## 1.3 Data Augmentation

| Method | Settings |
| -----  | -------- |
| Random Resized Crop | Default, (224, 224) |
| Random Horizontal Flip | Probability: 0.2 |
| Normalize| Mean: [0.485, 0.456, 0.406], Standard Deviation: [0.229, 0.224, 0.225] |


<br />  
<br />  

# Train the model
## 2.1 Download pretrained models(trained on ImageNet)
1. download [convnext_xlarge](https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth) pretrained model from [facebook research](https://github.com/facebookresearch/ConvNeXt)
2. **move them to  ./weights/**
## 2.2 train convnext_xlarge
Commands to train the model. **Note that** the first or second of the following three commands is enough!
1. If the pre-training model has been downloaded and put in the weights folder, run the following command:
	```
	python train.py --weights=''
	```
2. If the pre-trained model is not downloaded, but the model is allowed to download automatically by the internet, then run the following command：
	```
	python train.py --online_weight --weights='' 
	```
3. After training n epochs, the break point continues to train，then run the following command：

	```
	python train.py --pretra --weights='weights/best_model.pth'
	```
## 2.3 Get the trained model
After training the model, the files best_model.pth and latest_model.pth will be generated and saved in the weights folder
```
weights (The weight file for the model)
|   ├── convnext_xlarge_22k_224.pth
|   ├── best_model.pth (This file is generated after running train.py)
|   ├── latest_model.pth (This file is generated after running train.py)
```
<br />  
<br />  

# Test the  model
## 3.1 Commands to Test the model
```
	python predict_batch.py --weights='weights/best_model.pth'
```
## 3.2 Documents you need to submit
phase1.txt, pre_test.txt, and phase2.txt will be generated, and phase1.txt and phase2.txt need to be submitted to obtain the competition results
```
│── phase1.txt (This file is generated after running predict_batch.py)
│── pre_test.txt (This file is generated after running predict_batch.py)
│── phase2.txt (This file is generated after running predict_batch.py)
```

