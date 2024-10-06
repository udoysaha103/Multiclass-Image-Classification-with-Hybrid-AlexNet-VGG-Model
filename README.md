
# Sports Classification Using Deep Learning

This project implements a deep learning model to classify images of different sports into 100 categories. The model was trained using a Kaggle dataset and built from scratch, drawing inspiration from popular architectures like AlexNet and VGG. The aim is to create a robust image classification model that achieves high accuracy and generalization across a variety of sports images.



## Dataset

The dataset used for training the model is sourced from [Kaggle's Sports Classification dataset](https://www.kaggle.com/datasets/gpiosenka/sports-classification). It contains:

- 100 different sports categories.
- Training, validation, and test sets split as provided in the dataset.


### Dataset Format

- The dataset is stored in a CSV format with two important columns:
  - `filepaths`: Paths to the images.
  - `labels`: Labels representing different sports categories.


### Preprocessing

- Images are resized to 224x224 pixels.
- Augmentation techniques like horizontal flipping, zoom, and rotation are applied to the training set to make the model more generalizable.


## Model Architecture

The model consists of several convolutional layers followed by max pooling and fully connected layers. Key highlights of the architecture:

![ModelArchitecture](https://github.com/user-attachments/assets/935a0a7f-9dc4-4d82-ac73-86c084e01108)





## Results

### Model Performance

The final performance of the trained model on the test dataset:

- **Training Accuracy**: Approximately 98% - 99.99%
- **Validation Accuracy**: Approximately 83% - 86%
- **Test Accuracy**: Approximately 84.20% - 86.20%


### Evaluation Metrics

- **Confusion Matrix**: A confusion matrix is used to visualize the true and predicted labels of the model across all sports categories.

    ![download](https://github.com/user-attachments/assets/6db5d273-78f1-40a7-9319-ea0bf824eb2f)

- **Classification Report**: A classification report is used to analyze the precision, recall and f1-score of the model across all sports categories.
    ```bash
                            precision    recall  f1-score   support
               air hockey       0.83      1.00      0.91         5
          ampute football       1.00      1.00      1.00         5
                  archery       0.80      0.80      0.80         5
            arm wrestling       1.00      1.00      1.00         5
             axe throwing       1.00      0.80      0.89         5
             balance beam       0.83      1.00      0.91         5
            barell racing       1.00      1.00      1.00         5
                 baseball       1.00      1.00      1.00         5
               basketball       0.67      0.80      0.73         5
           baton twirling       0.40      0.40      0.40         5
                bike polo       1.00      0.80      0.89         5
                billiards       1.00      1.00      1.00         5
                      bmx       1.00      0.40      0.57         5
                  bobsled       0.80      0.80      0.80         5
                  bowling       1.00      0.40      0.57         5
                   boxing       1.00      1.00      1.00         5
              bull riding       0.67      0.80      0.73         5
           bungee jumping       0.60      0.60      0.60         5
             canoe slamon       0.83      1.00      0.91         5
             cheerleading       1.00      0.20      0.33         5
        chuckwagon racing       1.00      1.00      1.00         5
                  cricket       0.83      1.00      0.91         5
                  croquet       1.00      0.80      0.89         5
                  curling       1.00      1.00      1.00         5
                disc golf       0.83      1.00      0.91         5
                  fencing       0.75      0.60      0.67         5
             field hockey       0.71      1.00      0.83         5
       figure skating men       1.00      1.00      1.00         5
     figure skating pairs       0.80      0.80      0.80         5
     figure skating women       0.71      1.00      0.83         5
              fly fishing       0.67      0.80      0.73         5
                 football       0.83      1.00      0.91         5
         formula 1 racing       1.00      1.00      1.00         5
                  frisbee       0.50      0.20      0.29         5
                     gaga       1.00      0.60      0.75         5
             giant slalom       0.71      1.00      0.83         5
                     golf       1.00      0.80      0.89         5
             hammer throw       1.00      1.00      1.00         5
             hang gliding       1.00      1.00      1.00         5
           harness racing       0.62      1.00      0.77         5
                high jump       1.00      1.00      1.00         5
                   hockey       0.83      1.00      0.91         5
            horse jumping       0.62      1.00      0.77         5
             horse racing       1.00      1.00      1.00         5
       horseshoe pitching       1.00      1.00      1.00         5
                  hurdles       0.83      1.00      0.91         5
        hydroplane racing       1.00      0.60      0.75         5
             ice climbing       0.67      0.80      0.73         5
             ice yachting       1.00      0.80      0.89         5
                 jai alai       0.83      1.00      0.91         5
                  javelin       0.57      0.80      0.67         5
                 jousting       0.83      1.00      0.91         5
                     judo       0.80      0.80      0.80         5
                 lacrosse       0.67      0.80      0.73         5
              log rolling       1.00      0.60      0.75         5
                     luge       1.00      0.40      0.57         5
        motorcycle racing       0.71      1.00      0.83         5
                  mushing       1.00      0.80      0.89         5
            nascar racing       1.00      1.00      1.00         5
        olympic wrestling       1.00      1.00      1.00         5
             parallel bar       1.00      1.00      1.00         5
            pole climbing       0.75      0.60      0.67         5
             pole dancing       1.00      0.80      0.89         5
               pole vault       1.00      0.60      0.75         5
                     polo       1.00      1.00      1.00         5
             pommel horse       0.71      1.00      0.83         5
                    rings       1.00      0.80      0.89         5
            rock climbing       0.71      1.00      0.83         5
             roller derby       1.00      0.80      0.89         5
       rollerblade racing       1.00      1.00      1.00         5
                   rowing       1.00      0.80      0.89         5
                    rugby       0.80      0.80      0.80         5
          sailboat racing       1.00      1.00      1.00         5
                 shot put       0.75      0.60      0.67         5
             shuffleboard       1.00      1.00      1.00         5
           sidecar racing       0.80      0.80      0.80         5
              ski jumping       1.00      0.80      0.89         5
              sky surfing       0.80      0.80      0.80         5
                skydiving       0.71      1.00      0.83         5
            snow boarding       0.50      0.20      0.29         5
        snowmobile racing       0.50      0.80      0.62         5
            speed skating       1.00      1.00      1.00         5
          steer wrestling       1.00      0.60      0.75         5
           sumo wrestling       0.80      0.80      0.80         5
                  surfing       0.75      0.60      0.67         5
                 swimming       1.00      0.80      0.89         5
             table tennis       0.80      0.80      0.80         5
                   tennis       0.67      0.80      0.73         5
            track bicycle       1.00      1.00      1.00         5
                  trapeze       0.71      1.00      0.83         5
               tug of war       0.83      1.00      0.91         5
                 ultimate       1.00      0.80      0.89         5
              uneven bars       1.00      1.00      1.00         5
               volleyball       1.00      0.60      0.75         5
            water cycling       1.00      0.80      0.89         5
               water polo       0.71      1.00      0.83         5
            weightlifting       0.83      1.00      0.91         5
  wheelchair basketball       1.00      1.00      1.00         5
        wheelchair racing       0.80      0.80      0.80         5
          wingsuit flying       0.57      0.80      0.67         5

                 accuracy                           0.84       500
                macro avg       0.86      0.84      0.83       500
             weighted avg       0.86      0.84      0.83       500
    ```


## Improvements
Future work may involve:

- Implementing advanced techniques like transfer learning to improve accuracy.
- Data augmentation to increase the diversity of the training set.
- Hyperparameter tuning for optimizing model performance.



## Badges

Open access to anyone!

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)



## Contribute

Contributions are always welcome!




## 🚀 About Me

This is Udoy Saha. I am tech enthusiast, problem solver.

[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://udoysaha.com/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/udoysaha103/)


## Feedback

If you have any feedback, please reach out to me at udoysaha103@gmail.com.

