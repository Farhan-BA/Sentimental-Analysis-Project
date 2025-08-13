# Sentiview: A Streamlit-Powered Sentiment Analyzer

![Project Status](https://img.shields.io/badge/status-complete-green)
![Python Version](https://img.shields.io/badge/python-3.9+-blue)
![Framework](https://img.shields.io/badge/framework-Streamlit-ff4b4b)

A user-friendly web application built entirely in Python that performs real-time sentiment analysis on user-provided text. This project serves as a clear and straightforward example of how to deploy a complete machine learning model without needing JavaScript or complex web development frameworks.

---

### Table of Contents

* [About The Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [License](#license)
* [Acknowledgments](#acknowledgments)

---

## About The Project

Sentiview provides an interactive interface to instantly classify text as **Positive** or **Negative**. It leverages a classic machine learning pipeline (TF-IDF and Logistic Regression) and wraps it in a simple, elegant web app using the Streamlit framework.

### Built With

This project was built with the following technologies:

* [Python](https://www.python.org/)
* [Streamlit](https://streamlit.io/)
* [Scikit-learn](https://scikit-learn.org/)
* [NLTK](https://www.nltk.org/)
* [Pandas](https://pandas.pydata.org/)

---

## Getting Started

Follow these steps to get a local copy up and running.

### Prerequisites

You must have Python 3.8+ installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

### Installation

1.  **Clone the Repository (or create a project folder)**
    If this were on GitHub, you would clone it. For now, ensure you have a project folder containing:
    * `app.py`
    * `train_model.py`
    * `IMDB Dataset.csv`

2.  **Create and Activate a Virtual Environment**
    It is highly recommended to use a virtual environment to manage dependencies. Open your terminal in the project directory and run:

    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**
    Install all required packages using pip:
    ```bash
    pip install streamlit pandas scikit-learn joblib nltk
    ```

4.  **Train the ML Model (One-Time Step)**
    Before launching the app, you need to generate the model files. Run the training script:
    ```bash
    python train_model.py
    ```
    This command will create `sentiment_model.joblib` and `tfidf_vectorizer.joblib` in your directory.

---

## Usage

With the setup complete, you can now launch the web application.

1.  **Run the Streamlit App**
    In your terminal (with the virtual environment activated), execute the following command:
    ```bash
    streamlit run app.py
    ```

2.  **Interact with the App**
    Your web browser will automatically open a new tab with the Sentiview application. Enter any text into the text area and click the "Analyze Sentiment" button to see the prediction and confidence score.

---

## License

This project is distributed under the MIT License. See `LICENSE.txt` for more information. (Note: A `LICENSE.txt` file would need to be added).

---

## Acknowledgments

* Thanks to the creators of the IMDB Reviews Dataset.
* Inspiration from the Streamlit community.
