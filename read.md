## Scenario

A customer has approached the company to build a custom algorithm for detecting
suspicious activities in real-time on CCTV footage. The decision has been made to use an
off-the-shelf deep learning architecture for the required detector. However, data to train the
customer’s desired detector is not readily available.

## Questions

### Fine-Tuning vs. Fully Training a Model

* Describe the advantages and disadvantages of fine-tuning an off-the-shelf deep
  learning model compared to fully training a model from scratch for detecting
  suspicious activities in real-time on CCTV footage.
  * Fine-tuning involves adapting a pre-trained model, initially trained on a large dataset, for a specific task. This process is commonly employed because the pre-trained model already comprehends the low-level features (such as edges, textures, and shapes in images) of the modality (such as text, images, or audio) it was trained on. Consequently, even with a very small dataset, the model can be effectively fine-tuned to perform well on a new task.
    Advantages and Disadvantages of Fine-Tuning vs. Training from Scratch
    #### Fine-Tuning an Off-the-Shelf Model
    #### Advantages:
    -	Fine-tuning is particularly advantageous when working with a small dataset. Since the pre-trained model already understands basic features, it can adapt to the new task with less data [1]. This is especially beneficial if there is insufficient CCTV footage of suspicious activities.
    -	The model starts with pre-learned features, making the training process much quicker, often requiring fewer epochs compared to training from scratch.
    -	Fine-tuning demands fewer computational resources because the model has already undergone extensive training during the initial phase. This is particularly useful in training within a limited resource embedded environment.
    -	The model can transfer the features it has learned to the new task, often resulting in improved performance and better generalization. The pre-trained model must have learned the low-level features of a particular modality so that the fine-tuning phase can focus on training with the application-level dataset.
    #### Disadvantages:
    -	When using a pre-trained model, the architecture and weights of the lower layers are usually fixed, offering little flexibility for modification. However, it is possible to add or modify the final layers to adjust the model for a new purpose.
    -	If the pre-trained features are not well-suited to the new task, there is a risk of overfitting to the fine-tuning dataset [3]. Additionally, if the dataset for fine-tuning is too small and contains too few examples of suspicious activities, the model might overfit to these specific examples rather than generalizing to unseen footage.
    Training a Model from Scratch
    #### Advantages:
    -	Full Customization: Training from scratch allows for complete control over the model architecture, enabling the design of a model tailored specifically to the task and dataset. For suspicious activity detection, the model can be designed to perform a very specific task and excel on a custom-sourced dataset [4].
    -	Better Suitability for Specific Tasks: A model trained from scratch can be optimized to capture the unique characteristics of the specific task, potentially leading to better performance.
    -	No Dependency on Pre-Trained Models: There is no reliance on the availability or quality of pre-trained models, providing more flexibility in the model’s architecture.
    #### Disadvantages:
    -	High Computational Cost: Training a model from scratch requires significant computational resources and time, especially for large datasets [2].
    -	Large Dataset Requirement: Achieving high performance typically requires a large, well-annotated dataset, which may not always be available [2].
    -	Longer Development Time: The process of training, validating, and tuning a model from scratch is time-consuming and requires extensive experimentation.
    #### Conclusion
    -	**Fine-Tuning:** Ideal for scenarios where computational resources and time are limited, and a suitable pre-trained model is available. It is particularly effective when the dataset is small or when rapid deployment is needed.
    -	**Training from Scratch:** Best suited for highly specific tasks where full customization is necessary, and sufficient computational resources and a large dataset are available.
      For detecting suspicious activities on CCTV footage, fine-tuning an off-the-shelf model may provide a good balance between performance and practicality unless a very specific or novel type of suspicious activity requires a custom model from scratch.

#### References
1.	[Fine-tuning vs From Scratch: Do Vision & Language Models Have Similar Capabilities?](https://aclanthology.org/2022.lrec-1.161/)
2.	[Improved Fine-Tuning by Better Leveraging Pre-Training Data](https://arxiv.org/abs/2111.12292)
3.	[Training vs. Fine-tuning: What is the Difference?](https://encord.com/blog/training-vs-fine-tuning/)
4.	[Fine-Tuning vs Full Training vs Training from Scratch](https://www.analyticsvidhya.com/blog/2024/06/fine-tuning-vs-full-training-vs-training-from-scratch/#:~:text=Training%20from%20scratch%20is%20flexible,particular%20tasks%20with%20limited%20data.)


### Finding a Suitable Off-the-Shelf Model

* Outline how you might approach the task of finding a suitable off-the-shelf model
  architecture for the suspicious activity detection task. What criteria would you
  consider to ensure the model can perform real-time detection on CCTV footage?

  * To approach the task of finding a suitable off-the-shelf model architecture for suspicious activity detection, one could start by considering models designed for human action recognition (HAR). These models can be fine-tuned to meet the specific requirements of detecting suspicious activities.
    
    #### Identify the Task Requirements:
    Identify the specific task. Suspicious activity is a broad category that includes various aspects such as crime detection, fall detection, and weapon detection. It is important to clarify the client's problem statement to proceed with the selection of the dataset and model.
    
    #### Identify the Hardware Requirements:
    When selecting hardware for suspicious activity detection, it’s crucial to balance performance and runtime efficiency. High-performance HAR models typically require substantial computational resources and large amounts of RAM. However, for real-time surveillance scenarios, the model should be optimized to run efficiently on embedded devices. This ensures that the system can operate continuously and reliably in various environments without the need for extensive infrastructure.
    
    For high-performance requirements, GPUs or TPUs can be considered to handle the computational load. Since this is a real-time scenario, hardware specifically designed for the use case can be utilized. For example, for clients of VCA Technology, it’s important to consider hardware specifications such as CPU, GPU, and supported RAM of VCA servers or VCA AI cameras before choosing models. Other hardware options, like NVIDIA Jetson or Google Coral, can be considered if the use case demands their use.
    
    #### Model Selection Criteria:
    
    Effective processing of video input is crucial for recognizing temporally varying events, such as human actions. Therefore, models that extract spatiotemporal features from videos to identify action patterns are essential. The architecture we choose should be capable of modeling the temporal sequence along with spatial information.
    
    **Convolutional Neural Networks (CNNs):** CNNs are effective in modeling low- to mid-level features of image data and can generalize better than earlier handcrafted methods for action recognition, such as Histograms of Oriented Gradients (HOG), Histograms of Oriented Flow, and SIFT. Object detection models designed for low computational resources, like YOLOvX-tiny or MobileNetVx [4], can be used, but their application is mainly limited to detection. Specific modifications and custom additional layers are required to perform action classification.
    
    **Two-Stream CNN:** The two-stream CNN is a popular architecture for human action recognition in videos. This approach uses two separate streams to capture different types of information: spatial and temporal. The spatial stream processes the RGB frames of the video to capture appearance information, focusing on static features such as the background and objects in each frame [7]. A standard CNN, like VGG or ResNet, is typically used to extract these spatial features. The temporal stream processes optical flow between consecutive frames to capture motion, which is crucial for understanding actions.
    
    **3D CNNs:** Unlike 2D CNNs, which apply filters over spatial dimensions (height and width), 3D CNNs apply filters over three dimensions: height, width, and time. This allows the network to learn spatiotemporal features directly from the video frames [6]. Although 3D CNNs can outperform 2D CNNs and two-stream CNNs, they require a larger number of parameters to model both spatial and temporal features effectively, leading to higher computational costs, which can be a drawback for small embedded devices.
    
    **Long Short-Term Memory (LSTM) Networks:** LSTMs are a type of recurrent neural network (RNN) designed to capture temporal dependencies in sequential data. For action recognition, LSTMs process sequences of video frames to learn the temporal dynamics of actions. LSTM Fusion combines LSTMs with other models, such as CNNs, to leverage both spatial and temporal features. This typically involves a two-stream approach where a CNN extracts spatial features from individual frames, and an LSTM processes these features to capture temporal dynamics. LSTM fusion models generally outperform single-stream LSTM models by providing a more comprehensive understanding of the video data.
    
    **Transformer-Based Models:** Recently, transformer-based action recognition models that use self-attention mechanisms to capture temporal relationships among neighboring frames have been shown to outperform existing off-the-shelf models [6]. However, they require higher computational power. While these models excel in accuracy, their computational demands can be a challenge for real-time applications on embedded devices.
    
    **Trade-Off Between Real-Time Efficiency and Performance:**
    In real-time scenarios, there is often a trade-off between efficiency and performance. Models that achieve higher accuracy, such as 3D CNNs or transformers, typically require more computational resources [6], which may not be suitable for devices with limited processing power. More efficient architectures like two-stream CNNs might offer better real-time performance but possibly at the expense of some accuracy. Striking the right balance between model performance and real-time efficiency is crucial, especially for embedded systems.
    
    However, advancements in hardware, such as the NVIDIA Jetson series, are making it increasingly feasible to deploy high-performance models, like transformers, on embedded devices with appropriate optimizations.
    
    **Recent Research for Real-Time Human Activity Detection:**
    
    Recent research on real-time human activity recognition on embedded devices has highlighted that latency bottlenecks have been due to the Optical Flow Extractor used in the pipeline of the two-stream HAR model [3]. They have introduced a new RT-HARE system that, unlike traditional methods relying on Optical Flow (OF) extraction, uses IMFE to directly extract motion features from raw video frames. IMFE leverages the parallel processing capabilities of the GPU and Deep Learning Accelerator (DLA) to enhance efficiency further. Integrated into the RT-HARE system, IMFE enables real-time human action recognition at 30 frames per second. Despite its efficiency, IMFE maintains high recognition accuracy, making it suitable for embedded applications.

#### References:
1. [VCA Servers](https://vcatechnology.com/products/vcaservers)
2. [VCA AI Cameras](https://vcatechnology.com/products/ai-cameras)
3. [Real-Time Human Action Recognition on Embedded Platforms](https://arxiv.org/abs/2409.05662)
4. [Mobilenets: Efficient convolutional neural networks for mobile vision applications](https://arxiv.org/abs/1704.04861)
5. [VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking](https://arxiv.org/abs/2303.16727)
6. [Spatio-Temporal FAST 3D Convolutions for Human Action Recognition](https://arxiv.org/abs/1909.13474)
7. [Two-stream convolutional networks for action recognition in videos](https://arxiv.org/abs/1406.2199)    

