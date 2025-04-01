# Pretrained Models Experiment Learnings

- Not using pretrained weights provides much worse performance than using pretrained weights, where we do not perform any additional finetuning.

*Aim:* the aim of this experiment is to learn about and implement variations of transfer learning. As the first experiment in this line of experiments, a secondary goal is to develop reusable modules and a pipeline, in addition to data collection mechanisms, and model performance metrics.

*Methods:* We implemented this experiment using the [PyTorch](https://pytorch.org/) framework for deep learning. We wrote a `train_model()` function to abstract the training process of the model, a `test_model()` function to abstract the testing (validation) process of the model, and a `run_pretrained_model_experiment()` function to serve as a wrapper function to set the model's pre-training layers, and to train and test the model. Throughout this process, we implemented numerous functions to collect data, namely the training losses and accuracies, and the test accuracy. 