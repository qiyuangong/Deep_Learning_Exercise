# Deep Learning Exercise

Examples, Notebook, materials and suggestions for Deep Learning Exercise and materials.

`*` means recommended.

## Roles and tasks (First of all)

`*` Before your started. Pls read [AI Career Pathways: Put Yourself on the Right Track](https://workera.ai/candidates/report/), recommended by Andrew NG and YC etc. It will let you know:

1. Roles and tasks in AI ecosystem
2. Your current role
3. Target role (Skill set and tasks)

[中文版](README_CH.md)

## Knowledge Required

### Basic knowledge and Applications

1. Python and Numpy
2. Basic Neural Network components:
    - Loss functions
    - Layers & operations (FC, Relu, CNN, RNN etc)
    - Optimizations (SGD, adam)
3. Deep Learning Frameworks
    - PyTorch, Keras, TensorFlow
    - BigDL and Analytics Zoo for big data
4. Examples in different areas
    - Computer Vision (CV)
    - Natural Language Processing (NLP)
    - Reinforcement Learning (RL)
    - Generative Adversarial Networks (GAN)
5. Transfer Learning & Fine-tune
    - Computer Vision (CV)
    - Natural Language Processing (NLP)
6. Serving Trained models (Into production)
    - TensorFlow & PyTorch Serving
    - KubeFlow
    - Analytics-Zoo Web & Cluster Serving

**After talking with serveral people, who are learning deep learning by themselves. I found that in this stage, learning too much without a correct direction is very inefficient. To avoid going into wrong directions, I highly recommend them to go over Standford CS231n by Feifei Li. This course will give you an overview & basic knowledge of deep learning, and let you know the mapping between problem & solutions.**

**Tips:** Focus on applications and examples. Search with Google, and try several solutions if possible.

**How to:** Choose a DL framework, and learn with examples. DIY like building with LeGo. At this stage, [Kaggle](https://www.kaggle.com/) and [Google colab](https://colab.research.google.com/) will be your best playground.

### Advanced knowledge

1. Machine Learning
2. Math:
    - Numerical Computation
    - Line algebra
    - Probability and Information Theory
3. Detailed deep learning knowledge in different areas
    - Computer Vision (CV)
    - Natural Language Processing (NLP)
    - Reinforcement Learning (RL)
4. Algorithms in Deep Learning
5. Efficient methods & hardware for Deep Learning
    - Distall, low-precision, compression & quantization etc
    - Methods for training & inference
    - Hardwares for training & inference

**Tips:** Pls focus on one of them, i.e., become expert in one area (or a few area) rather than know everything but not good at any of them. Pay more attention on Deep Learning papers rather than blogs.

**How to:** Choose a most interesting or relevant (with your work) area, and dig in. Then, keep tracking this area and become an expert.

## Materials & Resources

### Python & Numpy Exercise

1. `*` [CS231n Numpy Tutorial](http://cs231n.github.io/python-numpy-tutorial/)
2. [numpy 100 exercises](https://github.com/rougier/numpy-100)
3. [numpy exercises](https://github.com/Kyubyong/numpy_exercises)

## Kaggle Notebook

1. [Alien vs. Predator images](https://www.kaggle.com/pmigdal/alien-vs-predator-images) [Notebook](https://github.com/qiyuangong/Deep_Learning_Exercise/blob/master/kaggle/alien-vs-predator/Transfer_learning_with_PyTorch.ipynb)

## Deep Learning Exercise & Examples

### Analytics-Zoo Examples

1. [Analytics-Zoo Streaming Object Detection & Text Classifcation Scala Example](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/streaming)
2. [Analytics-Zoo Streaming Object Detection & Text Classifcation Python Example](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/streaming)
3. [Analytics-Zoo Image Classification with Redis & Redis Streams](https://github.com/qiyuangong/image_classification_redis)
4. [Analytics-Zoo OpenVINO ResNet_v1_50 Image Classifcation Scala Example](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/vnni/openvino)
5. [Analytics-Zoo OpenVINO ResNet_v1_50 Image Classifcation Python Example](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/vnni/openvino)
6. Analytics-Zoo OpenVINO mobilenet_v1 Object detection Python Example (WIP)
7. Analytics-Zoo OpenVINO Inception_v3 Image Classifcation Python Example (WIP)

### TensorFlow & Keras Examples

1. [Handwritten Recognization: minist](https://github.com/qiyuangong/Deep_Learning_Exercise/tree/master/jupyter_notebook/tensorflow/image_classification/mnist)
2. [TensorFlow pre-trained Image Classification models](https://github.com/tensorflow/models/tree/master/research/slim)
3. [TensorFlow Image Classification Preprocessing](https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/preprocessing_factory.py#L47)
4. [TensorFlow pre-trained Object Detection models](https://github.com/tensorflow/models/tree/master/research/object_detection)
5. [TensorFlow IMDB](https://github.com/qiyuangong/Deep_Learning_Exercise/blob/master/jupyter_notebook/tensorflow/tutorials/imdb.ipynb)

### OpenVINO Examples

[OpenVINO optimizes and loads TensorFlow pre-trained models](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html):

1. [OpenVINO load TensorFlow resnet_v1_50](https://github.com/qiyuangong/Deep_Learning_Exercise/blob/master/document/OpenVINO%20Resnet_v1_50.md)
2. [OpenVINO load TensorFlow inception_v3](https://github.com/qiyuangong/Deep_Learning_Exercise/blob/master/document/OpenVINO%20Inception_v3.md)
3. [OpenVINO load TensorFlow vgg_19](https://github.com/qiyuangong/Deep_Learning_Exercise/blob/master/document/OpenVINO%20VGG_19.md)
4. [OpenVINO load TensorFlow mobilenet_v1](https://github.com/qiyuangong/Deep_Learning_Exercise/blob/master/document/OpenVINO%20Mobilenet_v1_224.md)

### PyTorch Examples

1. `*` [DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
2. [PyTorch Tutorials](https://github.com/pytorch/tutorials/tree/master/beginner_source)

## Blog or Channels

1. Subscribe on [deeplearning.ai](https://www.deeplearning.ai/), it will send blog and recommended paper throught email.
2. Subscribe deep learning related topics on [Medium][https://medium.com/].
3. [Colah](https://colah.github.io/)

## Courses & Books

### Books

1. `*` [Deep Learning with Python](https://github.com/fchollet/deep-learning-with-python-notebooks) by François Chollet, [Notebook](https://github.com/qiyuangong/Deep_Learning_Exercise/tree/master/jupyter_notebook/tensorflow/deep-learning-with-python-notebooks)
2. `*` [Deep Learning](https://www.deeplearningbook.org/) by Ian Goodfellow and Yoshua Bengio and Aaron Courville.
3. [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen.
4. [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781491957653/) by We McKinney
5. [Linear Algebra and Learning from Data](https://math.mit.edu/~gs/learningfromdata/) by Gilbert Strang
6. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) by RichardS Sutton
7. [Mathematics for Machine Learning](https://mml-book.github.io/)

### Courses

1. `*` [Stanford CS231n: Convolutional Neural Networks for Visual Recognition, Spring 2017](http://cs231n.stanford.edu/syllabus.html), [Youtube](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk) [Summary](https://github.com/mbadry1/CS231n-2017-Summary)
2. [Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning](https://www.coursera.org/learn/introduction-tensorflow), [Code](https://github.com/qiyuangong/Deep_Learning_Exercise/tree/master/python/tensorflow/introduction-tensorflow) and [Notebook](https://github.com/qiyuangong/Deep_Learning_Exercise/tree/master/jupyter_notebook/tensorflow/introduction-tensorflow)
3. [Google Coding TensorFlow](https://www.youtube.com/playlist?list=PLQY2H8rRoyvwLbzbnKJ59NkZvQAW9wLbx)
4. [deeplearning.ai](https://www.deeplearning.ai/)
    - [Neural Networks and Deep Learning (Course 1 of the Deep Learning Specialization)](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)
    - [Improving deep neural networks: hyperparameter tuning, regularization and optimization (Course 2 of the Deep Learning Specialization)](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc)
    - [Structuring Machine Learning Projects (Course 3 of the Deep Learning Specialization)](https://www.youtube.com/watch?v=dFX8k1kXhOw&list=PLkDaE6sCZn6E7jZ9sN_xHwSHOdjUxUW_b)
    - [Convolutional Neural Networks (Course 4 of the Deep Learning Specialization)](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)
5. [Stanford CS230: Deep Learning | Autumn 2018](https://www.youtube.com/watch?v=PySo_6S4ZAg&list=PLoROMvodv4rOABXSygHTsbvUz4G_YQhOb)
6. [Stanford CS229: Machine Learning](https://www.youtube.com/watch?v=UzxYlbK2c7E&list=PLEBC422EC5973B4D8)
7. [MIT 18.065 Matrix Methods in Data Analysis, Signal Processing, and Machine Learning, Spring 2018](https://www.youtube.com/watch?v=Cx5Z-OslNWE&list=PLUl4u3cNGP63oMNUHXqIUcrkS2PivhN3k)
8. [deeplearning.ai](https://www.deeplearning.ai/)
    - [Neural Networks and Deep Learning (Course 1 of the Deep Learning Specialization)](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)
    - [Improving deep neural networks: hyperparameter tuning, regularization and optimization (Course 2 of the Deep Learning Specialization)](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc)
    - [Structuring Machine Learning Projects (Course 3 of the Deep Learning Specialization)](https://www.youtube.com/watch?v=dFX8k1kXhOw&list=PLkDaE6sCZn6E7jZ9sN_xHwSHOdjUxUW_b)
    - [Convolutional Neural Networks (Course 4 of the Deep Learning Specialization)](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)
9. [Stanford CS224U: Natural Language Understanding | Spring 2019](https://www.youtube.com/watch?v=tZ_Jrc_nRJY&list=PLoROMvodv4rObpMCir6rNNUlFAn56Js20)
10. [Stanford CS224N: Natural Language Processing with Deep Learning | Winter 2019](http://web.stanford.edu/class/cs224n/index.html), [Youtube](https://www.youtube.com/watch?v=8rXD5-xhemo&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z)
11. [Stanford CS234: Reinforcement Learning | Winter 2019](http://web.stanford.edu/class/cs234/index.html), [Youtube](https://www.youtube.com/watch?v=FgzM3zpZ55o&list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u)

## Reference

1. [Analytics-Zoo](https://github.com/intel-analytics/analytics-zoo)
2. [TensorFlow](https://www.tensorflow.org/)
3. [OpenVINO](https://software.intel.com/en-us/openvino-toolkit)
4. [Coursea](https://www.coursera.org)
5. [Google Football](https://github.com/google-research/football)
6. [Apache MXNet](https://mxnet.apache.org/)
7. [Apache Spark](https://spark.apache.org/)
8. [deeplearning.ai](https://www.deeplearning.ai/)
9. [Keras](https://keras.io/)
10. [PyTorch](https://pytorch.org/)
11. [Kaggle](https://www.kaggle.com/)
12. [Google Colab](https://colab.research.google.com/)
13. [BigDL](https://github.com/intel-analytics/BigDL)
14. [OpenVINO Open Model Zoo](https://github.com/opencv/open_model_zoo)
15. [AI Career Pathways: Put Yourself on the Right Track](https://workera.ai/candidates/report/)

## Other resources

1. [Machine_Learning_Study_Path中文](https://github.com/linxid/Machine_Learning_Study_Path)
2. [Machine Learning Systems Design](https://github.com/chiphuyen/machine-learning-systems-design)
3. [Microsoft AI education materials for Chinese students, teachers and IT professionals](https://github.com/microsoft/ai-edu)
