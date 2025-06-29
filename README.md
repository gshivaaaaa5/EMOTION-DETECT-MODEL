This repository contains a **deep learning project** to classify human facial emotions from grayscale images using a Convolutional Neural Network (CNN). It supports 7 emotions:  
**angry, disgust, fear, happy, sad, surprise, neutral**.

The project includes:  
- CNN training with data augmentation  
- Class imbalance handling via class weights  
- Evaluation on test data  
- Single image emotion prediction  
- Grad-CAM heatmap visualization to explain model predictions  
- Streamlit web app for easy interactive usING
## 🗂️ Repository Structure
FOLDER STRUCTURE
face-emotion-detection/
│
├── src/
│   ├── train\_model.py           # Model training script
│   ├── evaluate\_model.py        # Test dataset evaluation
│   ├── predict\_single.py        # Predict custom images
│   ├── gradcam.py               # Grad-CAM visualization code
│   └── convert\_model.py         # Convert Sequential to Functional model
│
├── app/
│   └── app.py                  # Streamlit web app
│
├── models/
│   ├── emotion\_model.h5
│   └── emotion\_model\_functional.h5
│
├── test-pic/                   # Custom test images (ignored by Git)
├── face-dataset/               # Training & test datasets (ignored by Git)
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE

 📦 Dataset

This project uses the **FER2013-style dataset** (7 emotion classes).  
**Note:** The dataset is **NOT included** in this repo because it’s large.

You can download the dataset here:  
- [FER2013 Kaggle Dataset](https://www.kaggle.com/datasets/msambare/fer2013)  
- Or upload your own face emotion images structured as:

face-dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
├── test/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/


Full Features of Face Emotion Detection with Grad-CAM
Accurate Emotion Classification
Detects 7 core human emotions (angry, disgust, fear, happy, sad, surprise, neutral) from grayscale face images with a CNN model trained on FER2013-style datasets.

Data Augmentation for Robustness
Uses image augmentation techniques like flipping, rotation, and zoom to make the model generalize better on real-world data.

Class Imbalance Handling
Calculates class weights to prevent the model from being biased toward more frequent emotions.

Grad-CAM Visualization
Generates heatmaps on images showing which parts of the face influenced the model’s predictions—super useful for interpretability and trust.

Custom Image Prediction
Users can feed any face image and get predicted emotion with confidence score.

Streamlit Web App Interface
An interactive UI that lets users upload images and see real-time emotion prediction with Grad-CAM overlay.

Model Conversion Scripts
Converts between sequential and functional Keras models to enable advanced features like Grad-CAM.

Training & Evaluation Pipeline
Easy-to-run scripts to train on datasets, validate, and test the model for consistent results.

Lightweight Grayscale Input
Uses grayscale 48x48 pixel images which makes it computationally efficient.

Model Checkpointing & Early Stopping
Saves best performing models and avoids overfitting by stopping training early if no improvement.

🚀 Usage Workflow
Prepare your Dataset
Structure your face emotion dataset into train/test folders with subfolders per emotion.

Train the Model
Run the training script to build your CNN model with augmented data and class weights.

Evaluate on Test Set
Check accuracy and loss on unseen test data to validate model performance.

Predict on New Images
Use the prediction script or the Streamlit app to input new face images and get emotion labels + confidence.

Visualize with Grad-CAM
Generate heatmaps to explain model decisions and validate if it’s focusing on meaningful facial areas.

Business Idea & Real-World Applications

This project isn’t just a neat ML trick — it can actually add serious value across industries. Here’s where it shines:

1. Customer Experience Enhancement
Integrate into retail or service kiosks to gauge customer emotions in real-time and tweak service accordingly.

Measure emotional responses to marketing campaigns or product demos.

2. Mental Health Monitoring
Use as an assistive tool in telehealth apps to track patient mood and flag concerning emotional trends.

3. Human-Computer Interaction (HCI)
Emotion-aware software that adapts responses or UI based on user’s emotional state.

Video games that change difficulty or narrative based on player emotions.

4. Security & Surveillance
Detect stress or aggression in sensitive locations to alert security personnel.

5. Education & E-Learning
Monitor student engagement and frustration levels to personalize learning experience.

6. Market Research & Feedback Analysis
Analyze customer reactions during focus groups or product testing without intrusive surveys.

7. Entertainment & Social Media
Apps that add emotion-based filters or generate content suggestions.

 Why This Project Rocks for Business
Provides real-time emotion insights that unlock customer understanding

Works on lightweight grayscale images — easy to deploy on devices with limited compute

Comes with explainability via Grad-CAM to build trust

Modular & extendable — can be integrated in apps, kiosks, healthcare platforms, or research tools

Can be offered as SaaS, embedded SDK, or analytics backend
## 📜 License
This project is licensed under the [MIT License](LICENSE).

## 🙌 Acknowledgements

* [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
* TensorFlow & Keras Teams
* Streamlit Framework
* OpenCV for image processing


## 📞 Contact

For questions, reach out to me on GitHub or \gshivadev@gmail.com.


# Thanks for checking out the project! 🚀

