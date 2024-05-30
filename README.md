# real-time-object-detection
real-time object detection systems by integrating natural language processing (NLP) techniques for generating descriptive textual information about detected objects. The proposed system combines the capabilities of object detection algorithms with NLP models to automatically generate human-readable descriptions of objects identified in a video stream. The implementation involves preprocessing the detected object labels, selecting or developing an appropriate NLP model, and integrating it seamlessly into the object detection pipeline.
# Front-end and Back-end
Flask serves as the backbone for our application, chosen for its ease of use and ability to respond quickly to real-time data. Flask also supports the deployment of our model with minimal setup, and for the frontend part, I have used HTML and CSS.
![image](https://github.com/saishankar912/real-time-object-detection/assets/154368009/58588a76-ed13-4291-822d-5ae290d610fa)
![image](https://github.com/saishankar912/real-time-object-detection/assets/154368009/bdbe62ce-36fb-4e74-8619-528cd3c3b5e7)
# Object decetetion and Description
I utilize YOLOV8 and OpenCV, both widely recognized for their robust capabilities in handling real-time image and video data processing. YOLOV8 allows for the implementation of complex machine-learning algorithms that can accurately detect objects, while OpenCV assists in the preprocessing of video streams to optimize the detection process.
For the generation of textual descriptions, the project employs a Convolutional Neural Network (CNN) trained on the Flickr dataset.
