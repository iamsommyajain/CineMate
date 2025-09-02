# Movie Recommender System

This is a hybrid movie recommender system using **Collaborative Filtering (SVD)** and **Content-Based Filtering (Cosine Similarity on genres)** built with Python.  
It utilizes the **MovieLens dataset** (`ml-latest-small`) and offers recommendations through a **Streamlit web app**.

---

## Features

- **Collaborative Filtering:** Personalized movie recommendations for users based on SVD.  
- **Content-Based Filtering:** Find movies similar to a given title using genres.  
- **Hybrid Recommendations:** Combine collaborative and content-based suggestions.  
- **Debugging:** Optional debug prints for genre matrix, cosine similarity, and top recommendations.

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd movie-recommender
```
### 2. Create a virtual environment and activate it

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download MovieLens dataset

Download ml-latest-small or ml-latest and place it in the project root:
```bash
movie-recommender/ml-latest-small/
```
## Training Models

Run the training script to build SVD and content-based models:
```bash
python -m src.train
```

This will save models to ./models.

Debug prints show genre matrix, unique combinations, and cosine similarity samples.

Testing Recommendations

Test SVD and content-based recommendations in the terminal:
```bash
python -m src.test_recommend
```
Running the Web App

Launch the Streamlit app:
```bash
streamlit run app.py
```

1. Enter a user ID to get collaborative filtering recommendations.
2. Enter a movie title to get content-based similar movies.
3. Debug messages will appear in the terminal if enabled.

## Dependencies

- Python 3.10+
- pandas
- numpy
- scikit-learn
- surprise
- joblib
- streamlit

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## License

This project is open-source under the MIT License.

