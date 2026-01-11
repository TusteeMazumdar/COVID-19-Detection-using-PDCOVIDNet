
The early and accurate detection of COVID-19 is crucial for effective clinical intervention and disease containment. This project presents an image-based COVID-19 classification system using deep learning techniques applied to chest X-ray images. Inspired by the pDCOVIDNet architecture, the model leverages convolutional neural networks (CNNs) to learn discriminative features from medical images. The implementation is carried out using PyTorch, and the system is evaluated using standard performance metrics to assess its effectiveness.

## **INTRODUCTION:**
 Coronavirus Disease 2019 (COVID-19) has placed immense pressure on global healthcare systems. Chest X-ray imaging has emerged as a widely accessible diagnostic tool due to its low cost and rapid acquisition. However, manual interpretation of X-ray images is time-consuming and subject to observer variability.

Recent advancements in deep learning, particularly convolutional neural networks, have shown significant promise in automated medical image analysis. This project explores the application of CNN-based techniques for detecting COVID-19 from chest X-ray images in an automated and scalable manner.

## **Objective:**
* To design and implement a CNN-based model for COVID-19 detection from chest X-ray images
* To apply data preprocessing and augmentation techniques to improve generalization
* To evaluate model performance using standard classification metrics
* To analyze classification outcomes using confusion matrices and reports

## Dataset

- **Source:** Kaggle Dataset – *Chest X-ray Images*
- **Classes:** COVID-19, Pneumonia, Normal
- **Image Preprocessing:**
  - Images are resized to **224 × 224**
  - Grayscale conversion is applied
  - Pixel normalization is performed

  ## 4. Methodology

### 4.1 Data Preprocessing

The following preprocessing steps are applied to the chest X-ray images prior to model training:

- Image resizing to a fixed input dimension  
- Pixel normalization  
- Data augmentation techniques, including:
  - Rotation  
  - Flipping  
  - Scaling  
- Weighted sampling to address class imbalance  

---

### 4.2 Model Architecture

The model is inspired by the **Parallel-Dilated Convolutional Neural Network (pDCOVIDNet)** concept, which emphasizes multi-scale feature extraction from medical images.

The implemented architecture consists of:

- Multiple convolutional layers for feature extraction  
- Non-linear activation functions (**ReLU**)  
- Pooling layers for spatial dimensionality reduction  
- Fully connected layers for final classification  

This implementation is adapted for **academic experimentation and learning purposes** 

---

### 4.3 Training Configuration

- **Framework:** PyTorch  
- **Loss Function:** Cross-Entropy Loss  
- **Optimizer:** Adam  
- **Hardware:** GPU-enabled environment (Google Colab compatible)  
## 7. Tools and Technologies

- **Programming Language:** Python  
- **Deep Learning Framework:** PyTorch  

- **Libraries:**
  - NumPy  
  - Torchvision  
  - Scikit-learn  
  - Matplotlib  
  - Seaborn  

- **Development Platform:** Google Colab 

---

## 9. Future Work

- Extension to **multi-class classification** (COVID-19, Pneumonia, Normal)  
- Integration of **transfer learning models** such as ResNet and EfficientNet  
- Evaluation on **larger and more diverse datasets**  
- Deployment as a **research-oriented decision-support system**  

---

## 10. Ethical Considerations

This project is developed strictly for **academic and educational purposes**.  
It is **not a certified medical diagnostic system** and should **not be used for clinical decision-making**.

---

## 11. Reference Paper

Chowdhury, M. E. H., Rahman, T., Khandakar, A., et al.  
**PDCOVIDNet: A Parallel-Dilated Convolutional Neural Network Architecture for Detecting COVID-19 from Chest X-ray Images.**  
*Health Information Science and Systems*, 8(1), 2020.  
DOI: [10.1007/s13755-020-00119-3  ](https://doi.org/10.1007/s13755-020-00119-3)

---

## 13. Author

- **Name:** Tustee Mazumdar  
- **Course:** Computer Vision and Robotics  

