import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

class DeepLearningRecommender:
    def __init__(self, num_users, num_items, embedding_size=50, learning_rate=0.001):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        # User input
        user_input = Input(shape=(1,), name="user_input")
        user_embedding = Embedding(input_dim=self.num_users, output_dim=self.embedding_size, name="user_embedding")(user_input)
        user_vec = Flatten(name="user_flatten")(user_embedding)

        # Item input
        item_input = Input(shape=(1,), name="item_input")
        item_embedding = Embedding(input_dim=self.num_items, output_dim=self.embedding_size, name="item_embedding")(item_input)
        item_vec = Flatten(name="item_flatten")(item_embedding)

        # Concatenate user and item vectors
        concat = Concatenate(name="concatenate")([user_vec, item_vec])

        # Fully connected layers
        dense_1 = Dense(128, activation='relu', name="dense_1")(concat)
        dropout_1 = Dropout(0.3, name="dropout_1")(dense_1)
        dense_2 = Dense(64, activation='relu', name="dense_2")(dropout_1)
        dropout_2 = Dropout(0.3, name="dropout_2")(dense_2)

        # Output layer
        output = Dense(1, activation='linear', name="output")(dropout_2)

        # Build model
        model = Model(inputs=[user_input, item_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')

        return model

    def train(self, user_data, item_data, ratings, epochs=10, batch_size=32, validation_split=0.2):
        history = self.model.fit(
            [user_data, item_data], ratings, 
            epochs=epochs, batch_size=batch_size, 
            validation_split=validation_split
        )
        return history

    def predict(self, user_data, item_data):
        return self.model.predict([user_data, item_data])

    def save_model(self, filepath):
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

# Helper functions
def preprocess_data(data):
    """
    Preprocess raw data and split it into users, items, and ratings.
    Args:
        data: A DataFrame containing user, item, and rating information.
    Returns:
        user_data, item_data, ratings: Arrays for user IDs, item IDs, and ratings.
    """
    user_data = data['user_id'].values
    item_data = data['item_id'].values
    ratings = data['rating'].values
    return user_data, item_data, ratings

def create_sample_data(num_users, num_items, num_samples):
    """
    Create random sample data for testing purposes.
    Args:
        num_users: Number of unique users.
        num_items: Number of unique items.
        num_samples: Total number of samples to generate.
    Returns:
        user_data, item_data, ratings: Generated random data.
    """
    user_data = np.random.randint(0, num_users, size=(num_samples,))
    item_data = np.random.randint(0, num_items, size=(num_samples,))
    ratings = np.random.rand(num_samples)
    return user_data, item_data, ratings

if __name__ == "__main__":
    num_users = 1000
    num_items = 500
    embedding_size = 50
    learning_rate = 0.001
    epochs = 10

    # Generate random data for testing
    user_data = np.random.randint(0, num_users, size=(10000,))
    item_data = np.random.randint(0, num_items, size=(10000,))
    ratings = np.random.rand(10000)

    # Initialize the recommender model
    recommender = DeepLearningRecommender(num_users, num_items, embedding_size, learning_rate)

    # Training the model
    history = recommender.train(user_data, item_data, ratings, epochs=epochs)

    # Save the model
    model_filepath = "recommender_model.h5"
    recommender.save_model(model_filepath)

    # Load the model
    recommender.load_model(model_filepath)

    # Predicting for some user-item pairs
    predictions = recommender.predict(np.array([10, 20]), np.array([30, 40]))
    print(predictions)

    # Create additional sample data
    sample_user_data, sample_item_data, sample_ratings = create_sample_data(1000, 500, 10000)

    # Train model on sample data
    recommender.train(sample_user_data, sample_item_data, sample_ratings, epochs=5)

    # Predict new ratings after additional training
    new_predictions = recommender.predict(np.array([50, 60]), np.array([70, 80]))
    print(new_predictions)

# Model Improvements and Utilities

def evaluate_model(recommender, user_data, item_data, actual_ratings):
    """
    Evaluate the model on test data and return the Mean Squared Error (MSE).
    Args:
        recommender: Trained model for recommendation.
        user_data: Array of user IDs for evaluation.
        item_data: Array of item IDs for evaluation.
        actual_ratings: Array of actual ratings for evaluation.
    Returns:
        mse: Mean Squared Error of the model predictions.
    """
    predictions = recommender.predict(user_data, item_data)
    mse = np.mean((predictions - actual_ratings) ** 2)
    return mse

def split_data(user_data, item_data, ratings, test_size=0.2):
    """
    Split the data into training and testing sets.
    Args:
        user_data: Array of user IDs.
        item_data: Array of item IDs.
        ratings: Array of ratings.
        test_size: Proportion of data to use for testing.
    Returns:
        train_user_data, test_user_data, train_item_data, test_item_data, train_ratings, test_ratings: Split datasets.
    """
    num_samples = len(user_data)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    test_samples = int(test_size * num_samples)
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]

    train_user_data = user_data[train_indices]
    test_user_data = user_data[test_indices]
    train_item_data = item_data[train_indices]
    test_item_data = item_data[test_indices]
    train_ratings = ratings[train_indices]
    test_ratings = ratings[test_indices]

    return train_user_data, test_user_data, train_item_data, test_item_data, train_ratings, test_ratings

def tune_hyperparameters(recommender, user_data, item_data, ratings, learning_rates, embedding_sizes, epochs=5):
    """
    Perform hyperparameter tuning by testing different learning rates and embedding sizes.
    Args:
        recommender: Initial model to tune.
        user_data: Array of user IDs.
        item_data: Array of item IDs.
        ratings: Array of ratings.
        learning_rates: List of learning rates to test.
        embedding_sizes: List of embedding sizes to test.
        epochs: Number of epochs for each configuration.
    Returns:
        best_model: Model with the best performance.
        best_params: Dictionary containing the best hyperparameters.
    """
    best_mse = float('inf')
    best_model = None
    best_params = {}

    for lr in learning_rates:
        for emb_size in embedding_sizes:
            print(f"Training with learning rate {lr} and embedding size {emb_size}")
            recommender = DeepLearningRecommender(num_users=recommender.num_users, num_items=recommender.num_items, embedding_size=emb_size, learning_rate=lr)
            history = recommender.train(user_data, item_data, ratings, epochs=epochs)

            # Split data for validation
            train_user_data, test_user_data, train_item_data, test_item_data, train_ratings, test_ratings = split_data(user_data, item_data, ratings)

            # Evaluate performance
            mse = evaluate_model(recommender, test_user_data, test_item_data, test_ratings)
            print(f"Validation MSE for learning rate {lr}, embedding size {emb_size}: {mse}")

            if mse < best_mse:
                best_mse = mse
                best_model = recommender
                best_params = {"learning_rate": lr, "embedding_size": emb_size}

    print(f"Best hyperparameters: {best_params}, with MSE: {best_mse}")
    return best_model, best_params

def generate_recommendations(recommender, user_id, item_ids):
    """
    Generate recommendations for a given user based on a list of item IDs.
    Args:
        recommender: Trained model.
        user_id: ID of the user for whom recommendations are generated.
        item_ids: List of item IDs to rank.
    Returns:
        sorted_items: List of item IDs sorted by predicted rating in descending order.
    """
    user_data = np.array([user_id] * len(item_ids))
    item_data = np.array(item_ids)
    
    predictions = recommender.predict(user_data, item_data)
    sorted_indices = np.argsort(predictions.flatten())[::-1]
    sorted_items = [item_ids[i] for i in sorted_indices]
    
    return sorted_items

def print_recommendations(recommender, user_id, item_ids):
    """
    Print the top recommended items for a user.
    Args:
        recommender: Trained model.
        user_id: ID of the user.
        item_ids: List of item IDs to rank.
    """
    recommendations = generate_recommendations(recommender, user_id, item_ids)
    print(f"Top recommendations for user {user_id}:")
    for i, item_id in enumerate(recommendations[:10]):
        print(f"{i+1}. Item {item_id}")

# New evaluation metric - Root Mean Squared Error (RMSE)
def calculate_rmse(predictions, actual_ratings):
    """
    Calculate the Root Mean Squared Error between predictions and actual ratings.
    Args:
        predictions: Predicted ratings.
        actual_ratings: Actual ratings.
    Returns:
        rmse: Root Mean Squared Error.
    """
    mse = np.mean((predictions - actual_ratings) ** 2)
    rmse = np.sqrt(mse)
    return rmse

# Main flow
if __name__ == "__main__":
    # Define number of users and items for testing
    num_users = 1000
    num_items = 500
    embedding_size = 50
    learning_rate = 0.001
    epochs = 10

    # Create random data
    user_data, item_data, ratings = create_sample_data(num_users, num_items, 10000)

    # Split the data
    train_user_data, test_user_data, train_item_data, test_item_data, train_ratings, test_ratings = split_data(user_data, item_data, ratings)

    # Initialize and train the model
    recommender = DeepLearningRecommender(num_users, num_items, embedding_size, learning_rate)
    recommender.train(train_user_data, train_item_data, train_ratings, epochs=epochs)

    # Evaluate the model
    test_mse = evaluate_model(recommender, test_user_data, test_item_data, test_ratings)
    print(f"Test MSE: {test_mse}")

    # Hyperparameter tuning
    learning_rates = [0.001, 0.01]
    embedding_sizes = [50, 100]
    best_model, best_params = tune_hyperparameters(recommender, train_user_data, train_item_data, train_ratings, learning_rates, embedding_sizes)

    # Evaluate the best model on the test set
    best_test_mse = evaluate_model(best_model, test_user_data, test_item_data, test_ratings)
    print(f"Best model test MSE: {best_test_mse}")

    # Generate and print recommendations for a user
    user_id = 42
    item_ids = list(range(100))
    print_recommendations(best_model, user_id, item_ids)

    # Calculate RMSE on the test set
    predictions = best_model.predict(test_user_data, test_item_data)
    test_rmse = calculate_rmse(predictions, test_ratings)
    print(f"Test RMSE: {test_rmse}")

# Advanced Evaluation Metrics

def calculate_mae(predictions, actual_ratings):
    """
    Calculate the Mean Absolute Error (MAE) between predictions and actual ratings.
    Args:
        predictions: Predicted ratings.
        actual_ratings: Actual ratings.
    Returns:
        mae: Mean Absolute Error.
    """
    mae = np.mean(np.abs(predictions - actual_ratings))
    return mae

def evaluate_with_metrics(recommender, user_data, item_data, actual_ratings):
    """
    Evaluate the model on test data using various metrics: MSE, RMSE, MAE.
    Args:
        recommender: Trained recommendation model.
        user_data: Array of user IDs for evaluation.
        item_data: Array of item IDs for evaluation.
        actual_ratings: Array of actual ratings for evaluation.
    Returns:
        metrics: Dictionary containing MSE, RMSE, and MAE.
    """
    predictions = recommender.predict(user_data, item_data)
    mse = np.mean((predictions - actual_ratings) ** 2)
    rmse = np.sqrt(mse)
    mae = calculate_mae(predictions, actual_ratings)

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae
    }

    return metrics

# Saving and Loading Model Weights

def save_model_weights(recommender, filepath):
    """
    Save the model weights to a specified filepath.
    Args:
        recommender: Trained model.
        filepath: Path to save the weights.
    """
    recommender.model.save_weights(filepath)
    print(f"Model weights saved to {filepath}")

def load_model_weights(recommender, filepath):
    """
    Load model weights from a specified filepath.
    Args:
        recommender: Model instance to load the weights into.
        filepath: Path to load the weights from.
    """
    recommender.model.load_weights(filepath)
    print(f"Model weights loaded from {filepath}")

# Handling Cold Start Problem

def handle_cold_start_new_user(recommender, new_user_id, item_ids, global_avg_rating):
    """
    Generate recommendations for a new user (cold start) by recommending top items based on global average ratings.
    Args:
        recommender: Trained model.
        new_user_id: ID of the new user.
        item_ids: List of item IDs to consider for recommendations.
        global_avg_rating: Global average rating across all items.
    Returns:
        recommended_items: List of item IDs recommended for the new user.
    """
    # In the absence of historical data, recommend items with higher than average ratings
    recommendations = generate_recommendations(recommender, new_user_id, item_ids)
    
    top_items = []
    for item_id in recommendations:
        if item_id > global_avg_rating:
            top_items.append(item_id)
        if len(top_items) >= 10:  # Limit to top 10 items
            break
    
    return top_items

def handle_cold_start_new_item(recommender, user_ids, new_item_id, global_avg_rating):
    """
    Generate recommendations for a new item (cold start) by predicting how existing users will rate the new item.
    Args:
        recommender: Trained model.
        user_ids: List of user IDs to consider for recommendations.
        new_item_id: ID of the new item.
        global_avg_rating: Global average rating across all users.
    Returns:
        sorted_user_predictions: List of user IDs sorted by predicted rating for the new item.
    """
    item_data = np.array([new_item_id] * len(user_ids))
    predictions = recommender.predict(user_ids, item_data)

    # Rank users by their predicted rating of the new item
    sorted_indices = np.argsort(predictions.flatten())[::-1]
    sorted_user_predictions = [user_ids[i] for i in sorted_indices if predictions[i] > global_avg_rating]

    return sorted_user_predictions

# Model Scalability and Batch Inference

def batch_predict(recommender, user_data, item_data, batch_size=512):
    """
    Perform batch prediction to handle large-scale inference.
    Args:
        recommender: Trained recommendation model.
        user_data: Array of user IDs.
        item_data: Array of item IDs.
        batch_size: Batch size for prediction.
    Returns:
        predictions: Array of predicted ratings.
    """
    num_samples = len(user_data)
    predictions = []

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_user_data = user_data[start_idx:end_idx]
        batch_item_data = item_data[start_idx:end_idx]
        batch_predictions = recommender.predict(batch_user_data, batch_item_data)
        predictions.append(batch_predictions)

    return np.concatenate(predictions)

def large_scale_evaluation(recommender, test_user_data, test_item_data, test_ratings, batch_size=512):
    """
    Evaluate the model on large datasets using batch inference.
    Args:
        recommender: Trained model.
        test_user_data: Array of user IDs for evaluation.
        test_item_data: Array of item IDs for evaluation.
        test_ratings: Array of actual ratings for evaluation.
        batch_size: Batch size for evaluation.
    Returns:
        metrics: Dictionary containing MSE, RMSE, and MAE for the large-scale test set.
    """
    predictions = batch_predict(recommender, test_user_data, test_item_data, batch_size=batch_size)
    mse = np.mean((predictions - test_ratings) ** 2)
    rmse = np.sqrt(mse)
    mae = calculate_mae(predictions, test_ratings)

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae
    }

    return metrics

# Extended Model Performance Monitoring

class ModelPerformanceMonitor:
    def __init__(self):
        self.metrics_history = []

    def log_metrics(self, metrics):
        """
        Log the evaluation metrics.
        Args:
            metrics: Dictionary of evaluation metrics (MSE, RMSE, MAE).
        """
        self.metrics_history.append(metrics)
        print(f"Logged metrics: {metrics}")

    def get_metrics_history(self):
        """
        Retrieve the history of logged metrics.
        Returns:
            metrics_history: List of logged metrics over time.
        """
        return self.metrics_history

    def print_metrics_summary(self):
        """
        Print a summary of the logged metrics history.
        """
        print("Performance Metrics History:")
        for i, metrics in enumerate(self.metrics_history, 1):
            print(f"Evaluation {i}: MSE = {metrics['MSE']}, RMSE = {metrics['RMSE']}, MAE = {metrics['MAE']}")

# Main flow for evaluating large-scale recommendation

if __name__ == "__main__":
    # Configuration for large-scale evaluation
    num_users = 50000
    num_items = 10000
    embedding_size = 50
    learning_rate = 0.001
    epochs = 5

    # Create random large-scale data for testing
    user_data, item_data, ratings = create_sample_data(num_users, num_items, 1000000)

    # Split the data
    train_user_data, test_user_data, train_item_data, test_item_data, train_ratings, test_ratings = split_data(user_data, item_data, ratings)

    # Initialize the model
    recommender = DeepLearningRecommender(num_users, num_items, embedding_size, learning_rate)

    # Train the model
    recommender.train(train_user_data, train_item_data, train_ratings, epochs=epochs)

    # Perform large-scale evaluation using batch inference
    large_scale_metrics = large_scale_evaluation(recommender, test_user_data, test_item_data, test_ratings)
    print(f"Large-scale test set metrics: {large_scale_metrics}")

    # Save the model weights after training
    model_weights_filepath = "recommender_model_weights.h5"
    save_model_weights(recommender, model_weights_filepath)

    # Load the model weights for further usage
    load_model_weights(recommender, model_weights_filepath)

    # Monitor model performance over time
    monitor = ModelPerformanceMonitor()
    monitor.log_metrics(large_scale_metrics)
    monitor.print_metrics_summary()