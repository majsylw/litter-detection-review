# Trash detection - review of useful resources

A list of useful resources in the litter classification, detection or segmentation (mainly one class - garbage).
Created during the [detectwaste.ml](https://detectwaste.ml/) project.

## Contributing

Feel free to add **issue** with short description of new publication or create a **pull request** - add the new resource to the table or fill missing description.

## Relevant Repositories

- Datasets for litter recognition, detection and segmentation: [AgaMiko/waste-datasets-review](https://github.com/AgaMiko/waste-datasets-review)
- Detect waste in Pomerania project: [wimlds-trojmiasto/detect-waste](https://github.com/wimlds-trojmiasto/detect-waste/tree/main)

## Table of Contents

| Paper | Dataset | Classes | Task | Algorithms | Results | Code |
|:-----:|:-------:|:-------:|:----:|:----------:|---------|------|
| [Yang, M. et al., 2016](http://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf)| [Trashnet](https://github.com/garythung/trashnet/tree/master/data) | 6 | classification |  SVM<br>CNN (AlexNet)    | mAcc = 63% | [Github](https://github.com/garythung/trashnet) |
|[G. Mittal et al., 2016](https://dl.acm.org/doi/pdf/10.1145/2971648.2971731)| [GINI](https://github.com/spotgarbage/spotgarbage-GINI)  | 1 | localization |   GarbNet   |mAcc = 87.69%|[SpotGarbage app](https://github.com/KudaP/SpotGarbage) |
| [Awe, O. et al., 2017](http://cs229.stanford.edu/proj2017/final-reports/5226723.pdf) | augmented [Trashnet](https://github.com/garythung/trashnet/tree/master/data) | 3 |   detection   |     Faster R-CNN |  Landfill AP = 69.9% <br> Paper AP = 60.7%<br>Recycling = 74.4%<br>mAP = 68.3%  | :x: |
|    [M. S. Rad et al., 2017](https://arxiv.org/pdf/1710.11374.pdf)   |   self-created    | 25, but model works on 3 |    detection  | OverFeat-GoogLeNet architecture | cigarette Prec. = 63.2% <br>leaves Prec. = 77,35% |   :x:   |
| [C. Bircanoğlu et al., 2018](https://www.researchgate.net/publication/325626219_RecycleNet_Intelligent_Waste_Sorting_Using_Deep_Neural_Networks)  |[Trashnet](https://github.com/garythung/trashnet/tree/master/data)| 6 | classification |  ResNet50, MobileNet, InceptionResNetV2, DenseNet[121, 169, 201], Xception, RecycleNet| 95% Accuracy (DenseNet121) | :x: |
| [Aral, R.A. et al., 2018](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8622212&tag=1)  |[Trashnet](https://github.com/garythung/trashnet/tree/master/data) | 6 | classification |  MobileNet, Inception-V4, DenseNet[121, 169]| 95% Accuracy (DenseNet[121, 169]) | :x: |
| [Chu, Y. et al., 2018.](https://www.hindawi.com/journals/cin/2018/5060857/) | self-created | 5 | classification | AlexNet CNN, Multi hybird system (MHS) | 98.2% Accuracy (fixed orientation MHS) | :x: |
| [Wang, Y. al., 2018](https://www.researchgate.net/publication/329039476_Autonomous_garbage_detection_for_intelligent_urban_management) | self-created | 1 | detection |   Fast R-CNN   |     AP = 89% | [Github](https://github.com/Nerdyvedi/Garbage-Detection) |
| [Liu, Y. et al., 2018](https://iopscience.iop.org/article/10.1088/1742-6596/1069/1/012032/pdf) | self-created | 1 | detection | YOLOv2  |   Acc = 89.71%      | :x: |
| [Fulton, M. et al., 2019](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8793975) | [Trash-ICRA19](https://conservancy.umn.edu/handle/11299/214366) | 3 | detection | YOLOv2, Tiny-YOLO, Faster RCNN, SSD| Faster RCNN mAP=81 | :x: |
| [Carolis, B.D. et al., 2020](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9122693&tag=1) | self-created  | 4 |  detection | YOLOv3 | mAP@50 = 59.57% | :x: |
|[Proença, P.F. et al., 2020](https://arxiv.org/pdf/2003.06975.pdf)| [TACO](http://tacodataset.org/) | 60, but model was tested on 10, and 1 | segmentation | Mask RCNN |1-class mAP = 15.9% <br>10-class mAP = 17.6%| [Github](https://github.com/pedropro/TACO) |
|[Wang, T. et al., 2020](https://www.mdpi.com/1424-8220/20/14/3816)| [MJU-Waste](https://github.com/realwecan/mju-waste/) | 1 | segmentation | FCN, PSPNet, CCNet, DeepLabv3 | TACO mPP - 96.07%<br>MJU-Waste mPP = 97.14%  | :x: |
| [Hong, J. et al., 2020](https://arxiv.org/pdf/2007.08097.pdf) | [TrashCan 1.0](https://conservancy.umn.edu/handle/11299/214865) | 4 | segmentation, detection | Mask RCNN, Faster RCNN | Faster RCNN mAP=34.5, Mask R-CNN mAP=30.0 | :x: |

## Papers

![Sorting](https://upload.wikimedia.org/wikipedia/commons/4/4e/Garbage_dump_site.svg)

### Classification

-  Yang, M. et al., Classification of Trash for Recyclability Status, CS229 Project Report; Stanford University: Stanford, CA, USA, 2016. [[`pdf`](http://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf)]
     - dataset: The dataset spans six classes: glass, paper, cardboard, plastic, metal, and trash. The dataset consists of 2527 images (501 glass, 594 paper, 403 cardboard, 482 plastic, 410 metal, 137 trash), and it is annotated by category per image. The dataset consist of photographs of garbage taken on a white background; the different exposure and lighting were selected for each photo (mainly one object per photo). [[`download`]](https://github.com/garythung/trashnet/tree/master/data)
     - algorithm: Authors explore the SVM and CNN algorithms with the purpose of efficiently classifying garbage into six different recycling categories. They used an architecture similar to AlexNet but smaller in filter quantity and size.
     - results: The SVM achieved better results than the Neural Network. It achieved an accuracy of 63% using a 70/30 train/test data split. Neural network with a 70/30 train/test split achieved a testing accuracy of 27%.
     - code: [[`official code-lua-torch`]](https://github.com/garythung/trashnet)
- C. Bircanoğlu et al., RecycleNet: Intelligent Waste Sorting Using Deep Neural Networks, 2018 Innovations in Intelligent Systems and Applications (INISTA), pp. 1–7, 2018. [[`pdf`](https://www.researchgate.net/publication/325626219_RecycleNet_Intelligent_Waste_Sorting_Using_Deep_Neural_Networks)]
  - dataset: [Trashnet](https://github.com/garythung/trashnet/tree/master/data)
  - algorithm: Authors developed model named RecycleNet, which is carefully optimized deepconvolutional neural network architecture for classiﬁcation ofselected recyclable object classes. This novel model reduced thenumber of parameters in a 121 layered network from 7 millionto about 3 million.
  - results: For training without any pre-trained weights, Inception-Resnet, Inception-v4 outperformed allothers with 90% test accuracy. For transfer learning and ﬁne-tuning of weight parameters using ImageNet, DenseNet121 gavethe best result with 95% test accuracy.
  - code: :x:
- Aral, R.A. et al., Classification of trashnet dataset based on deep learning models, in Proceedings of the 2018 IEEE International Conference on Big Data (Big Data), Seattle, WA, USA; pp. 2058–2062, 2018.[[`pdf`](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8622212&tag=1)]
  - dataset: [Trashnet](https://github.com/garythung/trashnet/tree/master/data)
  - algorithm: In this study, authors tested well-known Deep Learning models to provide the most efficient approach. In this study, Densenet121, DenseNet169, InceptionResnetV2, MobileNet, Xception architectures were used for Trashnet dataset and __Adam__ and Adadelta were used as the optimizer in neural network models.
  - results: The most successful test accuracy rates were achieved with the fine-tuned Densenet-121 and Densenet-169 models. In the selection of the optimizer, Adam and Adadelta optimizers were tried with 100 epochs in InceptionResNetV2 model. As a result of this experiment, a higher test accuracy was obtained in the Adam optimizer.
  - code: :x:
- Chu, Y. et al., Multilayer hybrid deep-learning method for waste classification and recycling. Comput. Intell. Neurosci. 2018. [[`pdf`](https://www.hindawi.com/journals/cin/2018/5060857/)]
  - dataset: 5000 images with resolution 640 x 480 px were collected on plain, gray background. Each waste image is grouped with its counterpart numerical feature information as a data instance (50 categories), which is then manually labelled as either recyclable (4 main categories for recyclable part) or not.
  - algorithm: In this study, authors tested a hybrid CNN approach for waste classification (and AlexNet CNN to compare).
  - results: Both Multilayer hybrid system and CNN perform well when investigated items have strong image features. However, CNN performs poorly when waste items lack distinctive image features, especially for “other” waste. MHS achieves a significantly higher classification performance: the overall performance accuracies are 98.2% and 91.6%, (the accuracy of the reference model is 87.7% and 80.0%) under two different testing scenarios: the item is placed with fixed and random orientations.
  - code: :x:

### Detection

- G. Mittal et al., SpotGarbage: smartphone app to detect garbage using deep learning, in Proceedings of the 2016 ACM International Joint Conference on Pervasive and Ubiquitous Computing - UbiComp ’16, Heidelberg, Germany, 2016, pp. 940–945. [[`pdf`](https://dl.acm.org/doi/pdf/10.1145/2971648.2971731)]
  - dataset: Garbage in Images (GINI) dataset with 2561 images with unspecified resolution, 1496 images were annotated by bounding boxes (one class - trash). Bing Image Search API was used to create their dataset. [[`download`]](https://github.com/spotgarbage/spotgarbage-GINI)
  - algorithm: The authors utilize a pre-trained AlexNet, and their approach focuses on segmenting a pile of garbage in an image and provides no details about types of wastes in that segment. Their method is based on extracting image patches and combining their predictions, and therefore cannot capture the finer object boundary details.
  - results: GarbNet reached an accuracy of 87.69% for the task of the detection of garbage, but produced wrong predictions when in an image are detected objects similar to waste or when they are in the distance.
  - code: [[`official code-caffe`]](https://github.com/KudaP/SpotGarbage)
- Awe, O. et al., Smart trash net: Waste localization and classification. CS229 Project Report; Stanford University: Stanford, CA, USA, 2017. [[`pdf`](http://cs229.stanford.edu/proj2017/final-reports/5226723.pdf)]
  - dataset: Self-created augmented [Trashnet](https://github.com/garythung/trashnet/tree/master/data) dataset with 3 classes and 10,000 images with white background where individual waste was artificially combined.
  - algorithm: Faster R-CNN model pre-trained on PASCAL VOC dataset was used, which allows to recognize landfill, recycling, and paper.
  - results: The work presented reached a mAP of 68.30%.
  - code: :x:
- M. S. Rad et al., A Computer Vision System to Localize and Classify Wastes on the Streets, in Computer Vision Systems, 2017, pp. 195–204. [[`pdf`](https://arxiv.org/pdf/1710.11374.pdf)]
  - dataset: Self-created dataset with 25 different types of waste and 18 676 images at 640x480 pixels, collected from camera mounted on a vehicle, from Geneva streets, annotated by a bounding box around each waste.
  - algorithm: An open source implementation of OverFeat on Tensorflow was used with replacement of its classification architecture by GoogLeNet.
  - results: Precission above 60% only for two classes: most of wastes found in images were leaves (958 instances) and cigarette butts (69 instances of leaves and 394 bounding boxes on piles of leaves), only few examples of rest categories - 8 bottles, 5 cans, 6 goblets (finally grupped as others).
  - code: official :x:, but their work depends on [paper](https://arxiv.org/pdf/1506.04878.pdf) with [[`official code-caffe`]](https://github.com/wuyx/End-to-end-people-detection-in-crowded-scenes)
- Wang, Y. al., Autonomous garbage detection for intelligent urban management. In MATEC Web of Conferences, volume 232, page 01056. EDP Sciences, 2018. [[`pdf`](https://www.researchgate.net/publication/329039476_Autonomous_garbage_detection_for_intelligent_urban_management)]
  - dataset: Self-created garbage dataset with one class - garbage - and 816 images. Authors proposed a dataset fusion strategy, which integrates the garbage dataset with several other datasets of typical categories in urban scenes.
  - algorithm: Authors developed a Faster R-CNN open source framework with region proposal network and ResNet pre-trained on Dataset COCO.
  - results: They reached an accuracy of 89%. However, the model produced false positives when in an image there were also other objects in addition to waste.
  - code: [[`unofficial code-pytorch`]](https://github.com/Nerdyvedi/Garbage-Detection)
- Liu, Y. et al., Research on automatic garbage detection system based on deep learning and narrowband internet of things, in Journal of Physics: Conference Series, volume 1069, page 012032. IOP Publishing, 2018. [[`pdf`](https://iopscience.iop.org/article/10.1088/1742-6596/1069/1/012032/pdf)]
   - dataset: This paper collects data from existing decoration garbage database and self-built database. It uses 570 urban image data containing decoration garbage. The average size of the image is 420*400 pixels. In order to improve the generalization ability of the model, authors fuse and expand the data by combining the VOC2007 dataset.
    - algorithm: YOLOv2 is used. The architecture of the proposed neural network used MobileNet as feature extractor, and it is pretrained on ImageNet.
    - results: An accuracy of 89.71% is reported, but there was only one class in the dataset and they did not test on negative examples.
    - code: :x:
- Fulton, M. et al., Robotic Detection of Marine Litter Using Deep Visual Detection Models, 2019 International Conference on Robotics and Automation (ICRA), Montreal, QC, Canada, pp. 5752-5758, 2019. [[`pdf`](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8793975)]
  - dataset: The data (Trash-ICRA19 dataset) was sourced from the J-EDI dataset of marine debris. The videos that comprise that dataset vary greatly in quality, depth, objects in scenes, and the cameras used. They contain images of many different types of marine debris, captured from real-world environments, providing a variety of objects in different states of decay, occlusion, and overgrowth. Additionally, the clarity of the water and quality of the light vary significantly from video to video. These videos were processed to extract 5,700 images, which comprise this dataset, all labeled with bounding boxes on instances of trash, biological objects such as plants and animals, and ROVs. Trash detection data model has three classes, defined as follows:
    - Plastic: Marine debris, all plastic materials.
    - ROV: All man-made objects(i.e., ROV, permanent sensors, etc), intentionally placed in the environment.
    - Bio: All natural biological material, including fish, plants, and biological detritus. [[`download`]](https://conservancy.umn.edu/handle/11299/214366)
  - alghoritm: The four network architectures selected for this project were chosen from the most popular and successful object detection networks: YOLOv2, Tiny-YOLO, Faster RCNN with Inception v2, SSD with MobileNet v2. In the case of YOLO, the network was not only fine-tuned, but also trained using a transfer learning - authors  froze all but not the last several layers so that only the last few layers had their weights updated during the training process.
  - results: YOLOv2 - mAP=47.9, Tiny-YOLO - mAP=31.6, Faster RCNN with Inception v2 - mAP=81, SSD with MobileNet v2 - mAP=67.4. Faster R-CNN is the obvious choice purely from a standpoint of accuracy, but falters in terms of inference time. YOLOV2 strikes a good balance of accuracy and speed, while SSD provides the best inference times on CPU. 
  - code: :x:
- Carolis, B.D. et al., YOLO TrashNet: Garbage Detection in Video Streams, IEEE Conference on Evolving and Adaptive Intelligent Systems (EAIS), 2020. [[`pdf`](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9122693&tag=1)]
   - dataset: This paper construct a new dataset with 2714 images containing four classes: Garbage Bag, Garbage Dumpster, Garbage Bin and Blob (conglomerate of objects) and 1260 negative samples without any waste. The collection of the images of the dataset has been made using Google Images Download.
    - algorithm: YOLO TrashNet is used. In the experiment, authors train the last 53 layers of YOLOv3 on their dataset.
    - results: The mAP@50 was used as a metric to evaluate the error made by the network on unseen examples, and it reaches 59.57% (Garbage Bag - 40.56%, Garbage Dumpster - 64.57%, Garbage Bin - 65.01% and Blob - 68.15%).
    - code: :x:

### Segmentation

- Proença, P.F. et al., TACO: Trash Annotations in Context for Litter Detection. arXiv 2020. [[`pdf`](https://arxiv.org/pdf/2003.06975.pdf)]
  - dataset: 1500 images were collected from mainly outdoor environments such as woods, roads and beaches, and annotated by segmentation masks. There are 60 categories which belong to 28 super (top) categories. Additionaly images have background tag: water, sand, trash, vegetation, indor, pavement. [[`download`]](https://github.com/pedropro/TACO/tree/master/data)
  - algorithm: Authors adopted the Mask R-CNN implementation with Resnet-50 in a Feature Pyramid Network as a backbone with an input layer size of 1024×1024 px. Weights were started using Mask R-CNN weights trained on COCO dataset.
  - results: Authors provided results for a one-class and 10-class semantic segmentation task. They defined 3 metrics mAP besd on class score (maximum class probability), litter score and raitio score. The best value was achived for litter score on one-class atempt, and was equal 26.2%.
  - code: [[`official code-TensorFlow`]](https://github.com/pedropro/TACO) based on [Mask R-CNN by Matterport](https://github.com/matterport/Mask_RCNN)
- Wang, T. et al., A Multi-Level Approach to Waste Object Segmentation. Sensors 20(14), 2020. [[`pdf`](https://www.mdpi.com/1424-8220/20/14/3816)]
  - dataset: MJU-Waste consits of 2475 images taken indoor (usually one object per image), and uses a single class label for all waste objects. For each color image, authors provided the co-registered depth image captured using an RGBD camera. [[`download`]](https://drive.google.com/file/d/1o101UBJGeeMPpI-DSY6oh-tLk9AHXMny/view)
  - algorithm: Authors experimented with VGG16, ResNet-50 and ResNet-101 backbones in well known framewroks as FCN, PSPNet, CCNet, DeepLabv3. In addition authors proposed multi-level models, which improve the baseline performance (three levels - scene parsing for initial segmentation, object-level parsing for edges, pixel-level refinement).
  - results: Authors provided results for a two-class (waste vs. background) semantic segmentation task. Mean pixel precision on TACO equals 96.07%, on MJU-Waste - 97.14%, with DeepLabv3-ML and ResNet-101 backbone.
  - code: official :x:, but their work depends on [CNN-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch/) and [CRF](https://github.com/lucasb-eyer/pydensecrf/)
- Hong, J. et al., rashCan: A Semantically-Segmented Dataset towards Visual Detection of Marine Debris. arXiv 2020. [[`pdf`](https://arxiv.org/pdf/2007.08097.pdf)]
  - dataset: 7 212 images, which contain observations of trash, ROVs, and a wide variety of undersea flora and fauna, were collected. The annotations in this dataset take the format of instance segmentation annotations: bitmaps containing a mask marking which pixels in the image contain each object. The imagery in TrashCan is sourced from the J-EDI (JAMSTEC E-Library of Deep-sea Images) dataset, curated by the Japan Agency of Marine Earth Science and Technology (JAMSTEC). This dataset contains videos from ROVs operated by JAMSTEC since 1982, largely in the sea of Japan. The dataset has two versions, TrashCan-Material and TrashCan-Instance, corresponding to different object class configurations.[[`download`]](https://conservancy.umn.edu/handle/11299/214865)
  - algorithm: Authors adopted:
    -  for object detection: Faster R-CNN with a ResNeXt-101-FPN backbone; the model was trained on a pre-trained model with COCO dataset with an Nvidia Titan XP for 10, 000 iterations.
    -  for instance segmentation task: Mask R-CNN with X-101-FPN; the model was initialized with weights from the COCO dataset and trained for 15, 000 iterations using an Nvidia Titan V.
  - results: 
    |Metod | Type | AP | AP<sub>50</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
    |:---: | :---: | :---: | :---: | :---: | :---: | :---:  |
    |Faster R-CNN | instance | 30.0 | 55.3 | 29.4 | 31.7 | 48.6  |
    |Faster R-CNN | material | 28.2 | 54.0 | 25.6 | 28.7 | 41.8  |
    | Mask R-CNN | instance | 34.5 | 55.4 | 38.1 | 36.2 | 51.4  |
    | Mask R-CNN | material | 29.1 | 51.2 | 27.8 | 30.2 | 40.0  |
  - code: :x:, but authors claimed to use [Detectron2](https://github.com/facebookresearch/detectron2)
