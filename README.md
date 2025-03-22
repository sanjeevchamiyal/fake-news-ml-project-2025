# Fake News Detection Using Machine Learning

## Overview
This project is a **Fake News Detection System** that classifies news articles as **real or fake** using **machine learning models**. It is built using **Python**, trained with various classifiers, and deployed on **Streamlit** for real-time user interaction.

## Features
- **Data Preprocessing**: Tokenization, stopword removal, lemmatization.
- **Feature Extraction**: TF-IDF Vectorization.
- **ML Models**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting.
- **Performance Metrics**: Accuracy, Precision, Recall, F1-score.
- **Deployment**: Interactive web app using **Streamlit**.

## Technologies Used
- **Python** (NumPy, Pandas, Seaborn, Matplotlib, Scikit-learn)
- **Machine Learning Models**
- **Streamlit** (for deployment)
- **TF-IDF Vectorization** (for text feature extraction)

## Installation & Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
   cd fake-news-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. **Upload or enter news text**.
2. **Click on 'Check'** to classify it as Real or Fake.
3. **View prediction result**.

## Model Evaluation
- Models trained and tested on a **labeled dataset**.
- Achieved **high accuracy** with **Gradient Boosting** and **Random Forest**.
- Results visualized using **Seaborn and Matplotlib**.

## Future Enhancements
- Implement **deep learning models** for better accuracy.
- Add **real-time web scraping** for automatic fact-checking.
- Integrate **explainability features** to understand model decisions.

## Contributors
- Aastha Sharma

## License
This project is open-source under the **MIT License**.

