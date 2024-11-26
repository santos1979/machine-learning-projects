# README: Multi-Label Pathology Classification Using Deep Learning

This project implements a **multi-label pathology classification system** using **chest X-ray images**. The pipeline includes **data preprocessing, model training, evaluation, and visualization** using Grad-CAM++. The goal is to classify eight chest pathologies with advanced deep learning techniques, including custom loss functions and augmentation strategies to handle class imbalance.

---

## **Features**

- **Multi-Label Classification**: Classifies pathologies such as Atelectasis, Cardiomegaly, Emphysema, etc.
- **Imbalanced Dataset Handling**:
  - Focal Loss with alpha and class weights.
  - Oversampling and undersampling to balance the training set.
- **Custom Data Augmentation**:
  - MixUp for minority class augmentation.
  - Dynamic transformations using `ImageDataGenerator`.
- **Explainability**:
  - Grad-CAM++ to highlight regions influencing the predictions.
  - Visualizations to verify model attention areas.
- **Dynamic Thresholding**:
  - Per-class thresholds for optimal F1-Score using Precision-Recall curves.
- **Efficient Training**:
  - Model training with callbacks like Early Stopping and ReduceLROnPlateau.
  - Multi-GPU support with TensorFlow's `MirroredStrategy`.

---

## **Workflow Overview**

### 1. **Data Preparation**
   - Consolidates image and metadata files.
   - Filters metadata for available images and normalizes column names.
   - Creates binary columns for each pathology and ensures no overlap between training, validation, and test sets.

### 2. **Data Balancing**
   - **Oversampling**: Augments underrepresented classes.
   - **Undersampling**: Reduces overrepresented classes to avoid bias.

### 3. **Data Augmentation**
   - **Minority Classes**: Aggressive augmentation (rotations, shifts, zooms).
   - **Majority Classes**: Mild augmentation for slight variability.
   - **MixUp**: Generates new samples by mixing images and labels.

### 4. **Model Architecture**
   - Base models supported: `InceptionV3`, `EfficientNetB3`, `DenseNet121`, and `ResNet50`.
   - **Custom Layers**: Adds Dense, Dropout, and BatchNormalization layers.
   - Optimized with **AdamW** optimizer and L2 regularization.

### 5. **Loss Function**
   - **Focal Loss**: Adjusts for class imbalance.
   - **Hybrid Loss**: Combines Focal Loss, Binary Crossentropy, and Hinge Loss.

### 6. **Evaluation**
   - Metrics: AUC-ROC, Precision, Recall, and F1-Score.
   - Dynamic thresholds optimize binary predictions for each class.

### 7. **Visualization**
   - Grad-CAM++ heatmaps for explainability.
   - Confusion matrices for each class.
   - Side-by-side visualization of augmented images.

---

## **Results**

- **Best AUC-ROC**: 0.81 for Pneumothorax.
- **F1-Score Range**: Achieved moderate performance across classes, balancing recall and precision.
- Grad-CAM++ provided insights into the model's focus areas for specific pathologies.

---

## **Usage**

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Run Training**
```bash
python train_model.py
```

### **3. Generate Visualizations**
```bash
python visualize_gradcam.py
```

### **4. Evaluate Model**
```bash
python evaluate_model.py
```

---

## **Directory Structure**

```
├── data/
│   ├── images_all/           # Consolidated image files
│   ├── Data_Entry_2017.csv   # Metadata file
├── models/
│   ├── final_model.h5        # Trained model
├── notebooks/
│   ├── data_preparation.ipynb
│   ├── training.ipynb
│   ├── visualization.ipynb
├── scripts/
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── visualize_gradcam.py
├── README.md
```

---

## **Future Improvements**

1. **Hyperparameter Optimization**: Automate learning rate, loss parameters, and augmentation strategies.
2. **Additional Pathologies**: Expand classification to more pathologies.
3. **Deployment**: Integrate with a REST API for real-time predictions.

---

## **Acknowledgments**

- Chest X-ray dataset: [NIH Chest X-rays](https://www.nhlbi.nih.gov/science/x-ray-dataset)
- Pretrained models: TensorFlow and Keras applications.
- Visualization techniques inspired by Grad-CAM++.

---

For questions or collaborations, please reach out via [GitHub Issues](https://github.com/santos1979). 🎉
