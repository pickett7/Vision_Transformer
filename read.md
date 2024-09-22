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
