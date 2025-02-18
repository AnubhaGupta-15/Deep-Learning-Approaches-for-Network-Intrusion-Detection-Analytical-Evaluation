# **Deep Learning Approaches for Network Intrusion Detection: Analytical Evaluation**

## **📌 Project Overview**
This project explores **deep learning-based intrusion detection systems (NIDS)** to improve cybersecurity by detecting malicious activities in network traffic. The study compares multiple deep learning models and their effectiveness in identifying network attacks.

## **🚀 Key Contributions**
✅ Developed & Evaluated **Five Deep Learning Models:**
- **Deep Neural Network (DNN)**
- **Convolutional Neural Network (CNN)**
- **Long Short-Term Memory (LSTM)**
- **Gated Recurrent Unit (GRU)**
- **Autoencoder (AE)**

✅ **Achieved Over 90% Accuracy with DNN**, outperforming traditional machine learning techniques.

✅ **Optimized Model Training:**
- Processed **125,000+ network traffic records** from the **NSL-KDD dataset**.
- Applied **Logistic Regression for feature selection**, eliminating **79 irrelevant features** to improve model efficiency.

✅ **Performance Analysis:**
- Compared **Adam vs. SGD optimizers**, achieving **10% to 25% accuracy improvements** over existing intrusion detection models.
- Demonstrated potential for **real-time cybersecurity applications**.

---

## **📂 Dataset: NSL-KDD**
- The **NSL-KDD dataset** is used for intrusion detection system (IDS) training and evaluation.
- It consists of **various network traffic records**, classified into normal and attack categories.
- More details: [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)

---

## **📌 Installation & Setup**
### **🔹 Prerequisites**
Ensure you have the following installed:
- **Python 3.8+**
- **TensorFlow / Keras**
- **NumPy, Pandas, Scikit-Learn**
- **Matplotlib / Seaborn** (for visualization)

### **🔹 Install Required Libraries**
```sh
pip install numpy pandas scikit-learn tensorflow keras matplotlib seaborn
```



### **🔹 Run the Model**
```sh
python main.py
```



**Accuracy Formula:**
```
ACCURACY = (TP + TN) / (TP + TN + FP + FN)
```

---

## **🔬 Methodology**
### **1️⃣ Data Preprocessing**
- **Feature Normalization:** Data was scaled between 0 and 1.
- **Categorical Encoding:** One-hot encoding was used for categorical attributes.
- **Feature Selection:** Logistic regression eliminated 79 irrelevant features, reducing feature space from 123 to 44 features.

### **2️⃣ Deep Learning Models**
- **DNN:** Fully connected neural network with 4 hidden layers**.
- **CNN:** Extracts spatial features from network traffic**.
- **LSTM:** Captures temporal dependencies, achieving**.
- **GRU:** Optimized for sequence-based intrusion detection**.
- **Autoencoder:** Unsupervised anomaly detection, achievin**.

---

## **🎯 Results & Insights**
- **DNN outperformed all models**, achieving the highest accuracy (**95% +**).
- **Adam optimizer consistently outperformed SGD**, improving model efficiency.
- The proposed models significantly improved **false positive rates** and **real-time detection capabilities**.

---

## **🌍 Future Work**
- **Implement additional datasets** (e.g., UNSW-NB15, CIC-IDS2017) to generalize findings.
- **Optimize hyperparameters** for enhanced performance.
- **Deploy the best-performing model in real-time NIDS systems**.

---

## **📜 License**
This project is released under the **MIT License**.

---

## **📞 Contact**
👩‍💻 **Anubha Gupta**  
📧 Email: [your.email@example.com](mailto:your.email@example.com)  
🔗 GitHub: [AnubhaGupta-15](https://github.com/AnubhaGupta-15)  
