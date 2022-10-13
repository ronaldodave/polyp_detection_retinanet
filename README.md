# Polyp Detection in Colonoscopy Image Using CNN With RetinaNet Architecture

Colorectal cancer always begins with the appearance of polyps in the colon which can turn into malignant tumors and cause cancer. Therefore, it is necessary to screen the large intestine using colonoscopy. However, according to studies, about 26% of polyps are missed during colonoscopy procedures. In this study, a Convolutional Neural Network (CNN) with RetinaNet architecture was implemented to detect the location of polyps in colonoscopy images automatically. Three types of backbone architecture are used on RetinaNet :
 - ResNet-50, 
 - ResNet-101
 - ResNet-152
 
 ![comparison table](https://raw.githubusercontent.com/ronaldodave/polyp_detection_retinanet/main/result/result_table.jpg)
 
 From the evaluation results, the best model based on the Intersection over Union (IoU) metric is the RetinaNet model (Backbone = ResNet-50) without augmentation data with a value of 0.8415. While the best model based on the Average Precision (AP) metric is RetinaNet (Backbone = ResNet-101) with data augmentation with values ​​AP25 = 0.9308, AP50 = 0.9039, AP75 = 0.6985.
 
## Source
[Fizyr - keras-retinanet Repository](https://github.com/fizyr/keras-retinanet)
[Kvasir SEG Dataset](https://datasets.simula.no/kvasir-seg/)
[Etis-Larib Dataset](https://polyp.grand-challenge.org/Databases/)
[CVC ClinicDB Dataset](https://polyp.grand-challenge.org/CVCClinicDB/)
