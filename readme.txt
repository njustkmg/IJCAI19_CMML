Code of Comprehensive Semi-Supervised Multi-Modal Learning.
For any question, please contact Yang Yang (yyang@njust.edu.cn); Ketao Wang (wangkt@lamda.nju.edu.cn) or De-Chuan Zhan (zhandc@lamda.nju.edu.cn).
Enjoy the code.

**************************** Requirement ****************************
#requirement Python3.6, PyTorch0.4.0

******************************* USAGE *******************************
code  ----- The main code of the algorithm. It takes data(img madality and text madality)/label as input.

--the data file should contain:
#'sample_img': img madality data
#'sample_coco_imgs.pkl': name list of img madality data
#'sample_coco_text.npy': text madality feature data
#'sample_coco_label.npy': data label

--the parameters
In the function of train:
#'cita': parameter of ¦Ä in huberloss

In the main:
#'Textfeaturepara': architecture of text feature network 
#'Textpredictpara': architecture of text predict network 
#'Imgpredictpara': architecture of img predict network 
#'Predictpara': architecture of attention predict network 
#'Attentionparameter': architecture of attention network
#'superviseunsuperviseproportion': ratio of supervise data to unsupervise data, 
                                   for example, '2,8' means proportion of supervise data is 20% and proportion of unsupervise data is 80%

--demo:
data/: dataset of COCO, there are 2 madalities including img madality and text madality
i.e.
python Deep_attention_strong_weak_train.py

***************************** REFERENCE *****************************
If you use this code in scientific work, please cite:
Yang Yang, Ke-Tao Wang, De-Chuan Zhan, Hui Xiong and Yuan Jiang. Comprehensive Semi-Supervised Multi-Modal Learning. In: Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI'19).
*********************************************************************
