from project import *

#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz"
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d6_coco17_tpu-32.tar.gz"

imagePath = "test/ts.jpg"
classFile = "coco.names"
#videoPath= "test/preview.mp4"
#videoPath= 0
threshold = 0.5

project = Project()
project.readClasses(classFile)
project.downloadModel(modelURL)
project.loadModel()
project.predictImage(imagePath, threshold)
#project.predictVideo(videoPath, threshold)