import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
import streamlit as st
import logging
import pickle
import os

# Setting up the logger
logging.basicConfig(
    filename='app.log',
    filemode='w',
    level=logging.DEBUG,
    format="%(asctime)s : %(name)s : %(levelname)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Load datasets
@st.cache_data
def fetch_data(filename):
    logging.debug(f"Fetching the {filename} data")
    fetch_dir = os.path.join(os.getcwd(), 'data')
    data_path = os.path.join(fetch_dir, f"{filename}.parquet")
    return pd.read_parquet(data_path)


# Prepare the data for Surprise library
def prepare_data(data):
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(data[['user_id', 'product_id', 'rating']], reader)
    return dataset


# Build and train the SVD model
def build_model(trainset):
    logging.debug("Training the model")
    model = SVD()
    model.fit(trainset)
    logging.info("Model successfully trained")
    
    # Save the model after training
    with open('svd_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    logging.info("Saved the trained model")
    
    return model


# Load the model if it already exists
def load_model():
    logging.debug("Loading model if already exists")
    if os.path.exists('svd_model.pkl'):
        with open('svd_model.pkl', 'rb') as model_file:
            return pickle.load(model_file)
    return None


# Evaluate the model on the test set
def evaluate_model(model, test_data):
    logging.debug("Evaluating model on test data")
    testset = prepare_data(test_data)
    testset = testset.build_full_trainset().build_testset()
    logging.info("Prepared the test set")
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    logging.info(f"Model RMSE = {rmse:.2f}")
    return rmse


# Get recommendations for a user
def get_recommendations(model, user_id, data, num_items=5):
    logging.debug("Getting recommendations for user")
    user_data = data[data['user_id'] == user_id]
    user_products = user_data['product_id'].unique()
    
    all_products = data['product_id'].unique()
    products_to_predict = [p for p in all_products if p not in user_products]
    
    predictions = []
    for product in products_to_predict:
        pred = model.predict(user_id, product)
        predictions.append((product, pred.est))
    
    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    return [product for product, _ in predictions[:num_items]]

# Streamlit app
st.title("Amazon Products Recommendation System")
st.write("<h3>Created By: Mohammed Misbahullah Sheriff</h3>", unsafe_allow_html=True)

# Load training and testing data
train_data = fetch_data('train')
test_data = fetch_data('test')
logging.info("Loaded train and test datasets")

# Prepare the training data for modeling
trainset = prepare_data(train_data)
trainset = trainset.build_full_trainset()
logging.info("Prepared the train set")

# Check if model exists, else train a new one
model = load_model()
if model is None:
    model = build_model(trainset)
logging.info("Successfully loaded the model")

# Evaluate the model on the test data
rmse = evaluate_model(model, test_data)

# Create a dropdown for user ID selection
unique_user_ids = train_data['user_id'].unique()
user_id = st.selectbox("Select User ID:", unique_user_ids)

if st.button("Get Recommendations"):
    recommendations = get_recommendations(model, user_id, train_data)
    st.write(f"Top 5 Recommendations for User:", user_id)
    st.write(recommendations)
    logging.info("Successfully recommended products for given user")