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

### Creating a Training Dataset

* Given that data to train the desired detector is not readily available, describe how you
  would put together a training dataset of sufficient size to train an off-the-shelf deep
  learning algorithm for suspicious activity detection.

* What factors affect the amount of data required for training the detector, and 
how would you determine the necessary quantity?
  * Several factors influence the amount of data required to train a detector effectively, particularly for tasks like suspicious activity detection.

    To ensure robustness and better generalization, a larger and more diverse dataset is typically recommended for suspicious activity recognition. A diverse dataset allows the model to learn a wider variety of patterns, reducing the likelihood of overfitting and improving its performance on new, unseen data.
    
    The factors that affect the amount of data required for training the detector include the complexity of the task, the variability of the training data, and the model’s capacity.
    
    **Task Complexity**
    Suspicious activity detection is a complex task that requires the model to understand variations in human behaviour. This complexity necessitates more data to capture these variations effectively.
    
    **Model Requirements**
    Complex models with higher capacity require a large amount of data to generalize well for suspicious activity detection. Deep learning models, in particular, tend to overfit on small datasets and thus need more data to improve their generalization. Simpler models can be trained on smaller datasets, but they typically offer less accuracy.
    
    **Data Distribution**
    The quality of the data is crucial for training complex models. Data with limited diversity in lighting conditions and backgrounds can hinder the model’s ability to generalize to unknown environments. Improper classification and labelling of diverse data can lead to poor classification performance. An imbalance in class distribution can bias the model and reduce its accuracy.
    
    To determine the necessary quantity of data, we can initially train the models for action recognition using existing datasets like UCF101 or CASIA. If these datasets are insufficient, we can gradually increase the size by collecting more data or using synthetic data generation.
    
    **Evaluation**
    Experimenting with different dataset sizes and cross-validating the models’ performance can help gauge whether the current data is sufficient or if more diversity and volume are needed.

* Detail the processes you would use to source and label the data in the required
  quantity for training the detector to operate effectively on real-time CCTV footage.
  * To gather the data needed for training a detector to effectively analyze real-time CCTV footage, several strategies can be employed, keeping in mind the privacy and security concerns that come with surveillance footage.

  First, we can leverage readily available open-source datasets that are suitable for CCTV applications. It’s important to ensure these datasets comply with privacy regulations and don’t involve any sensitive data that could be exposed during inference. Datasets like DCSAAS are ideal, and similar resources can be found on platforms like Kaggle. However, it is crucial to restrict data collection to fully open-source datasets to avoid legal and ethical issues.
  
  Additionally, if customers can provide footage from real-time scenarios sourced in-house, it would greatly enhance the dataset's relevance and accuracy. This real-world data can reflect specific environments and conditions that the detector will encounter.
  
  If in-house footage is not available, public video platforms like YouTube can be utilized to source videos relevant to the target actions or behaviours. This approach can supplement the dataset with diverse scenarios and activities, ensuring a broader understanding of potential suspicious behaviours.

  For labeling the dataset, third-party labeling services can be used to streamline the process and reduce the manual effort involved in annotating actions. These services can assist in tagging large volumes of data efficiently. Additionally, we can employ AI models for initial rough classification of the data, allowing for a faster preliminary labeling phase. This approach can significantly reduce the amount of manual annotation required. Afterward, we can focus on manually annotating any misclassified samples, making it easier to refine the dataset without starting from scratch.

### Training and Testing Procedure

* Outline the training and testing procedure you would use for the model. Include
  information on which metrics would be used to evaluate the model’s performance
  and explain why these metrics are important for the task.

  * The first step is to establish a clear training objective. Suspicious activity detection encompasses various topics, so it’s essential to specify the exact task the model will be used.
  If a pre-trained model for the desired task is unavailable, it’s advisable to begin by training the model with publicly available large datasets that are closely aligned with the tasks at hand.
  In our case, datasets like UCF101 [6] [7], NTURGB-D, and Kinetics [6] [7] can be used to train the model for general human action recognition. We can then acquire public datasets with actions more closely related to the specific task, such as UCFCrime [3] [4] or ShanghaiTech [5], which include video data of various abnormal activities like theft, loitering, and abuse.
  It is always best to obtain real-time CCTV footage from the client, which can be annotated later to create a custom dataset. Ensure that the dataset includes a variety of scenes, featuring both normal and abnormal activities, to avoid overfitting the model to a single scenario.[2]
  Instead of annotating from scratch, we can leverage AI models pre-trained for action recognition to initially annotate the data, then refine the labels for any misclassifications.
  #### **Data Preprocessing:**
  -  Extract frames from videos, resize them and normalize pixel values.
  -  Apply augmentation techniques like random cropping, rotation, horizontal flipping, brightness adjustment, and motion blur to simulate various environmental conditions (e.g., changes in lighting or camera angles).
  -  In practice, Human Action Recognition (HAR) systems may not always operate at high frame rates due to varying scenarios and camera configurations. Additionally, methods with higher latency might struggle to meet real-time constraints when frame rates are high. [1]
  During frame extraction, videos can be downsampled to 30, 15, 6, or 3 frames per second (FPS).
  #### **Model Selection:**
  - Split the dataset into training, validation, and test sets (e.g., 70% for training, 15% for validation, and 15% for testing). For smaller datasets, training and validation splits can be used, or if a larger dataset is available, an 80/20 train-test split may be appropriate.
  -	Define a clear training objective: for video-based HAR, this is to minimize the error between the recognized and true action labels.
  For classification tasks (e.g., suspicious vs. non-suspicious activities), use categorical cross-entropy.
  -	Use optimizers like Adam or SGD with scheduled learning rate decay to ensure efficient convergence.
  -	Implement early stopping based on validation performance (e.g., stop training if validation loss doesn’t improve for a set number of epochs).
  -	Use video clips or sequences of frames (e.g., 16-32 frames per clip) as input. Train the model with mini-batch gradient descent to handle larger datasets efficiently.
  #### **Post-Training:**
  -	Depending on the deployment platform, export the model in the appropriate format, such as PyTorch or ONNX.
  -	Further optimize the model using techniques like quantization or pruning.
  -	Utilize optimization tools such as TensorRT (for NVIDIA hardware) or OpenVINO (for Intel hardware) to accelerate inference.
  -	Optimize the model for edge devices without sacrificing accuracy.
  #### **Testing and Evaluation:**
  Evaluation metrics are critical in suspicious activity recognition, as they provide an objective measure of the model's performance, helping to gauge its reliability and effectiveness for real-world deployment. Key metrics such as accuracy, precision, recall, and F1 score are important for evaluating how well the model detects actual threats while minimizing false alarms. Additionally, benchmark scores for accuracy and F1 score are useful for comparing models and understanding their expected performance.
  #### Key Evaluation Metrics:
  #### Accuracy:
  **Definition:** The percentage of correctly identified activities (both suspicious and non-suspicious) out of the total activities.
  **Importance:** While accuracy provides an overall measure of performance, it can be misleading in imbalanced datasets, where normal activities far outnumber suspicious ones.
  **Example:** In a surveillance system, if 98% of activities are normal and only 2% are suspicious, a model could achieve 98% accuracy simply by classifying everything as normal, which isn’t helpful for detecting real threats.
  #### Precision:
  **Definition:** The proportion of true positives (correctly identified suspicious activities) out of all predicted suspicious activities.
  **Importance:** High precision ensures that most flagged events are genuinely suspicious, reducing false positives, which is crucial for preventing unnecessary interventions.
  **Example:** In a retail store, a high-precision model would minimize false alarms by not flagging regular shoppers as potential thieves.
  Precision = True Positives (TP) / (True Positives (TP) + False Positives (FP))
  •	Recall:
  o	Definition: Recall (or sensitivity) measures the proportion of actual suspicious activities that the model correctly identifies.
  o	Importance: High recall ensures the system detects most suspicious activities, minimizing false negatives (missed suspicious activities).
  o	Example: In a public space, high recall is vital for identifying threats like theft or violence, even at the cost of a few false positives.
  Recall = True Positives (TP) / (True Positives (TP) + False Negatives (FN))
  •	F1 Score:
  o	Definition: The harmonic mean of precision and recall, providing a balanced measure of the model’s performance.
  o	Importance: Since suspicious activity detection requires balancing precision and recall, the F1 score is crucial for assessing overall performance in real-world scenarios.
  F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
  o	Example: A surveillance system with a high F1 score would efficiently detect most threats while minimizing false alerts.
  Focusing on the F1 score provides a balanced view of the model’s ability to perform well across different types of errors, while precision and recall highlight specific strengths and weaknesses.


