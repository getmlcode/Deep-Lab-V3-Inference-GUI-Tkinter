# Deep-Lab-V3-Inference-GUI-Tkinter
Tkinter based GUI wrapper for semantic segmentation using deeplab V3 (Inference Only).

[ Download Pretrained Model ]( https://www.dropbox.com/sh/s7sx69pqjhrk0s4/AACXWCRd9JJ0zvcvDES9G3sba?dl=0 )  
  
Original code for inference runs only from command line and works only with tfrecords files. This one lets the user select image from their disk using a GUI and works with jpg and png formats.  

One can also use [ this file ]( https://github.com/getmlcode/Deep-Lap-V3-Inference-GUI-Tkinter/blob/master/semanticSegmentation.py ) to run from command line.  

# Dependencies  
Tensorflow 1.11.0  
tkinter 8.6  
numpy 1.15.3  
PIL 1.1.7  
json 2.0.9  
matplotlib 3.0.1 ( needed if running [ this file ]( https://github.com/getmlcode/Deep-Lap-V3-Inference-GUI-Tkinter/blob/master/semanticSegmentation.py ) from command line )  

# Demo
![](semSegDeepLab.gif)

# Reference
Uses resnet files from [here](https://github.com/sthalles/deeplab_v3/tree/master/resnet)
and network.py from [here](https://github.com/sthalles/deeplab_v3)  
[DeepLabV3](https://arxiv.org/pdf/1706.05587.pdf)

