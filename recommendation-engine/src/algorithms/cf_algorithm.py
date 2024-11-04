import numpy as np
from scipy.sparse import csr_matrix

class CollaborativeFiltering:
    def __init__(self, user_item_matrix, k=5):
        """
        Initialize the CollaborativeFiltering class.

        :param user_item_matrix: The user-item interaction matrix.
        :param k: Number of similar users/items to consider.
        """
        self.user_item_matrix = user_item_matrix
        self.k = k
        self.similarity_matrix = None
        self.normalized_matrix = None

    def normalize_matrix(self):
        """
        Normalize the user-item matrix by subtracting the mean rating for each user.

        This helps in handling variations in individual user rating scales.
        """
        user_means = np.mean(self.user_item_matrix, axis=1, keepdims=True)
        self.normalized_matrix = self.user_item_matrix - user_means

    def compute_similarity(self):
        """
        Compute the similarity matrix using cosine similarity between users.

        This is achieved by calculating the dot product between normalized user vectors.
        """
        if self.normalized_matrix is None:
            self.normalize_matrix()

        sim_matrix = self.normalized_matrix.dot(self.normalized_matrix.T)
        norms = np.linalg.norm(self.normalized_matrix, axis=1)
        self.similarity_matrix = sim_matrix / np.outer(norms, norms)
        np.fill_diagonal(self.similarity_matrix, 0)  # Set self-similarity to zero

    def predict(self, user_index, item_index):
        """
        Predict the rating a user would give to an item.

        :param user_index: Index of the user.
        :param item_index: Index of the item.
        :return: Predicted rating.
        """
        sim_scores = self.similarity_matrix[user_index]
        user_ratings = self.user_item_matrix[:, item_index]

        top_k_users = np.argsort(sim_scores)[-self.k:]
        top_k_sim_scores = sim_scores[top_k_users]
        top_k_ratings = user_ratings[top_k_users]

        weighted_sum = np.dot(top_k_sim_scores, top_k_ratings)
        sum_of_weights = np.sum(np.abs(top_k_sim_scores))

        if sum_of_weights == 0:
            return 0

        return weighted_sum / sum_of_weights

    def recommend(self, user_index, num_recommendations=5):
        """
        Recommend top N items for a given user.

        :param user_index: Index of the user.
        :param num_recommendations: Number of recommendations to generate.
        :return: List of recommended item indices.
        """
        unrated_items = np.where(self.user_item_matrix[user_index, :] == 0)[0]
        predictions = [self.predict(user_index, item) for item in unrated_items]
        recommended_items = np.argsort(predictions)[-num_recommendations:]
        return recommended_items

    def fit(self):
        """
        Fit the collaborative filtering model by computing the similarity matrix.
        """
        self.compute_similarity()

    def get_top_k_similar_users(self, user_index, k=None):
        """
        Get the top K similar users for a given user.

        :param user_index: Index of the user.
        :param k: Number of similar users to return.
        :return: List of indices of similar users.
        """
        if k is None:
            k = self.k

        user_similarities = self.similarity_matrix[user_index]
        top_k_similar_users = np.argsort(user_similarities)[-k:]
        return top_k_similar_users

    def get_user_ratings(self, user_index):
        """
        Get the ratings given by a user.

        :param user_index: Index of the user.
        :return: List of ratings by the user.
        """
        return self.user_item_matrix[user_index, :]

    def update_user_item_matrix(self, new_user_item_matrix):
        """
        Update the user-item matrix with a new matrix.

        :param new_user_item_matrix: The new user-item matrix.
        """
        self.user_item_matrix = new_user_item_matrix
        self.normalized_matrix = None  # Reset the normalized matrix

    def calculate_error(self, predicted_ratings, true_ratings):
        """
        Calculate the error between predicted and true ratings.

        :param predicted_ratings: Predicted ratings for a user.
        :param true_ratings: Actual ratings given by the user.
        :return: Error value.
        """
        return np.sqrt(np.mean((predicted_ratings - true_ratings) ** 2))

    def cross_validate(self, n_folds=5):
        """
        Perform cross-validation to evaluate the model.

        :param n_folds: Number of folds for cross-validation.
        :return: Average error across folds.
        """
        n_users = self.user_item_matrix.shape[0]
        fold_size = n_users // n_folds
        errors = []

        for fold in range(n_folds):
            test_indices = np.arange(fold * fold_size, (fold + 1) * fold_size)
            train_indices = np.setdiff1d(np.arange(n_users), test_indices)

            train_matrix = self.user_item_matrix[train_indices, :]
            test_matrix = self.user_item_matrix[test_indices, :]

            self.update_user_item_matrix(train_matrix)
            self.fit()

            fold_errors = []
            for user_idx in test_indices:
                user_ratings = test_matrix[user_idx, :]
                rated_items = np.where(user_ratings > 0)[0]
                predicted_ratings = np.array([self.predict(user_idx, item) for item in rated_items])

                fold_error = self.calculate_error(predicted_ratings, user_ratings[rated_items])
                fold_errors.append(fold_error)

            errors.append(np.mean(fold_errors))

        return np.mean(errors)

# Usage
if __name__ == "__main__":
    # Sparse matrix creation
    user_item_data = csr_matrix([
        [4, 0, 0, 5, 1],
        [5, 5, 4, 0, 0],
        [0, 0, 0, 2, 4],
        [3, 3, 0, 0, 5]
    ]).toarray()

    cf = CollaborativeFiltering(user_item_data, k=2)
    cf.fit()

    # Make predictions for a user and recommend items
    user_index = 0
    num_recommendations = 3
    recommendations = cf.recommend(user_index=user_index, num_recommendations=num_recommendations)
    print("Recommendations for user {}: {}".format(user_index, recommendations))

    # Cross-validate the model
    error = cf.cross_validate(n_folds=3)
    print("Cross-validation error:", error)

    def get_top_k_similar_items(self, item_index, k=None):
        """
        Get the top K similar items for a given item.

        :param item_index: Index of the item.
        :param k: Number of similar items to return.
        :return: List of indices of similar items.
        """
        if k is None:
            k = self.k

        item_similarities = self.similarity_matrix[:, item_index]
        top_k_similar_items = np.argsort(item_similarities)[-k:]
        return top_k_similar_items

    def item_based_similarity(self):
        """
        Compute the similarity matrix using cosine similarity between items.
        """
        if self.normalized_matrix is None:
            self.normalize_matrix()

        sim_matrix = self.normalized_matrix.T.dot(self.normalized_matrix)
        norms = np.linalg.norm(self.normalized_matrix.T, axis=1)
        self.similarity_matrix = sim_matrix / np.outer(norms, norms)
        np.fill_diagonal(self.similarity_matrix, 0)  # Set self-similarity to zero

    def predict_item_based(self, user_index, item_index):
        """
        Predict the rating a user would give to an item using item-based collaborative filtering.

        :param user_index: Index of the user.
        :param item_index: Index of the item.
        :return: Predicted rating.
        """
        sim_scores = self.similarity_matrix[item_index]
        item_ratings = self.user_item_matrix[user_index, :]

        top_k_items = np.argsort(sim_scores)[-self.k:]
        top_k_sim_scores = sim_scores[top_k_items]
        top_k_ratings = item_ratings[top_k_items]

        weighted_sum = np.dot(top_k_sim_scores, top_k_ratings)
        sum_of_weights = np.sum(np.abs(top_k_sim_scores))

        if sum_of_weights == 0:
            return 0

        return weighted_sum / sum_of_weights

    def recommend_item_based(self, user_index, num_recommendations=5):
        """
        Recommend top N items for a given user using item-based collaborative filtering.

        :param user_index: Index of the user.
        :param num_recommendations: Number of recommendations to generate.
        :return: List of recommended item indices.
        """
        unrated_items = np.where(self.user_item_matrix[user_index, :] == 0)[0]
        predictions = [self.predict_item_based(user_index, item) for item in unrated_items]
        recommended_items = np.argsort(predictions)[-num_recommendations:]
        return recommended_items

    def calculate_item_based_error(self, predicted_ratings, true_ratings):
        """
        Calculate the error between predicted and true ratings using item-based collaborative filtering.

        :param predicted_ratings: Predicted ratings for a user.
        :param true_ratings: Actual ratings given by the user.
        :return: Error value.
        """
        return np.sqrt(np.mean((predicted_ratings - true_ratings) ** 2))

    def cross_validate_item_based(self, n_folds=5):
        """
        Perform cross-validation to evaluate the item-based collaborative filtering model.

        :param n_folds: Number of folds for cross-validation.
        :return: Average error across folds.
        """
        n_users = self.user_item_matrix.shape[0]
        fold_size = n_users // n_folds
        errors = []

        for fold in range(n_folds):
            test_indices = np.arange(fold * fold_size, (fold + 1) * fold_size)
            train_indices = np.setdiff1d(np.arange(n_users), test_indices)

            train_matrix = self.user_item_matrix[train_indices, :]
            test_matrix = self.user_item_matrix[test_indices, :]

            self.update_user_item_matrix(train_matrix)
            self.item_based_similarity()

            fold_errors = []
            for user_idx in test_indices:
                user_ratings = test_matrix[user_idx, :]
                rated_items = np.where(user_ratings > 0)[0]
                predicted_ratings = np.array([self.predict_item_based(user_idx, item) for item in rated_items])

                fold_error = self.calculate_item_based_error(predicted_ratings, user_ratings[rated_items])
                fold_errors.append(fold_error)

            errors.append(np.mean(fold_errors))

        return np.mean(errors)

    def recommend_for_all_users(self, num_recommendations=5, item_based=False):
        """
        Generate recommendations for all users in the dataset.

        :param num_recommendations: Number of recommendations to generate for each user.
        :param item_based: Whether to use item-based collaborative filtering.
        :return: Dictionary mapping user indices to recommended item indices.
        """
        recommendations = {}
        for user_index in range(self.user_item_matrix.shape[0]):
            if item_based:
                recommendations[user_index] = self.recommend_item_based(user_index, num_recommendations)
            else:
                recommendations[user_index] = self.recommend(user_index, num_recommendations)

        return recommendations

    def update_ratings(self, user_index, item_index, new_rating):
        """
        Update the rating for a specific user-item pair.

        :param user_index: Index of the user.
        :param item_index: Index of the item.
        :param new_rating: New rating to assign.
        """
        self.user_item_matrix[user_index, item_index] = new_rating
        self.normalized_matrix = None  # Reset the normalized matrix

    def evaluate_model(self, metric='rmse', item_based=False):
        """
        Evaluate the collaborative filtering model based on a specified metric.

        :param metric: The evaluation metric to use. Currently supports 'rmse'.
        :param item_based: Whether to evaluate using item-based collaborative filtering.
        :return: The evaluation score.
        """
        if metric == 'rmse':
            errors = []
            for user_index in range(self.user_item_matrix.shape[0]):
                true_ratings = self.get_user_ratings(user_index)
                rated_items = np.where(true_ratings > 0)[0]

                if item_based:
                    predicted_ratings = np.array([self.predict_item_based(user_index, item) for item in rated_items])
                else:
                    predicted_ratings = np.array([self.predict(user_index, item) for item in rated_items])

                error = self.calculate_error(predicted_ratings, true_ratings[rated_items])
                errors.append(error)

            return np.mean(errors)
        else:
            raise ValueError(f"Unsupported evaluation metric: {metric}")

    def save_model(self, file_path):
        """
        Save the current model (user-item matrix and similarity matrix) to a file.

        :param file_path: The file path where the model will be saved.
        """
        np.savez(file_path, user_item_matrix=self.user_item_matrix, similarity_matrix=self.similarity_matrix)

    def load_model(self, file_path):
        """
        Load a saved model (user-item matrix and similarity matrix) from a file.

        :param file_path: The file path from which the model will be loaded.
        """
        data = np.load(file_path)
        self.user_item_matrix = data['user_item_matrix']
        self.similarity_matrix = data['similarity_matrix']
        self.normalized_matrix = None  # Reset the normalized matrix

    def get_recommendation_score(self, user_index, item_index, item_based=False):
        """
        Get the recommendation score for a specific user-item pair.

        :param user_index: Index of the user.
        :param item_index: Index of the item.
        :param item_based: Whether to use item-based collaborative filtering.
        :return: The recommendation score.
        """
        if item_based:
            return self.predict_item_based(user_index, item_index)
        else:
            return self.predict(user_index, item_index)

    def compare_user_based_item_based(self, user_index, item_index):
        """
        Compare the predicted rating for a user-item pair using both user-based and item-based collaborative filtering.

        :param user_index: Index of the user.
        :param item_index: Index of the item.
        :return: Dictionary with user-based and item-based prediction scores.
        """
        user_based_prediction = self.predict(user_index, item_index)
        item_based_prediction = self.predict_item_based(user_index, item_index)
        return {
            "user_based": user_based_prediction,
            "item_based": item_based_prediction
        }

    def get_similarity_matrix(self, item_based=False):
        """
        Retrieve the similarity matrix.

        :param item_based: Whether to return the item-based similarity matrix.
        :return: The similarity matrix (either user-based or item-based).
        """
        if item_based:
            if self.similarity_matrix is None:
                self.item_based_similarity()
            return self.similarity_matrix
        else:
            if self.similarity_matrix is None:
                self.compute_similarity()
            return self.similarity_matrix

    def personalized_recommendations(self, user_index, user_preference_weights, num_recommendations=5):
        """
        Generate personalized recommendations by adjusting the predicted ratings with user preference weights.

        :param user_index: Index of the user.
        :param user_preference_weights: Weights to adjust the recommendations based on user preferences.
        :param num_recommendations: Number of recommendations to generate.
        :return: List of personalized recommended item indices.
        """
        unrated_items = np.where(self.user_item_matrix[user_index, :] == 0)[0]
        predictions = np.array([self.predict(user_index, item) for item in unrated_items])
        
        # Adjust predictions with user preferences
        adjusted_predictions = predictions * user_preference_weights[unrated_items]
        recommended_items = np.argsort(adjusted_predictions)[-num_recommendations:]
        return recommended_items

    def incorporate_feedback(self, user_index, item_index, feedback_rating):
        """
        Incorporate user feedback by updating the user-item matrix with the new feedback rating.

        :param user_index: Index of the user.
        :param item_index: Index of the item.
        :param feedback_rating: New rating provided by the user.
        """
        self.user_item_matrix[user_index, item_index] = feedback_rating
        self.normalized_matrix = None  # Reset normalized matrix to incorporate feedback
        self.compute_similarity()  # Recompute similarity after feedback

    def batch_update_ratings(self, update_data):
        """
        Batch update ratings for multiple user-item pairs.

        :param update_data: List of tuples containing (user_index, item_index, new_rating).
        """
        for user_index, item_index, new_rating in update_data:
            self.update_ratings(user_index, item_index, new_rating)

    def generate_top_n_recommendations_for_all(self, num_recommendations=5, item_based=False):
        """
        Generate top N recommendations for all users and return a sorted list of recommendations for each user.

        :param num_recommendations: Number of recommendations per user.
        :param item_based: Whether to use item-based collaborative filtering.
        :return: A dictionary where keys are user indices and values are lists of recommended item indices.
        """
        recommendations = {}
        for user_index in range(self.user_item_matrix.shape[0]):
            if item_based:
                rec_items = self.recommend_item_based(user_index, num_recommendations)
            else:
                rec_items = self.recommend(user_index, num_recommendations)
            recommendations[user_index] = rec_items

        return recommendations

    def get_item_popularity(self):
        """
        Calculate the popularity of each item based on the number of users who have rated it.

        :return: A dictionary where keys are item indices and values are the count of users who rated the item.
        """
        item_popularity = np.sum(self.user_item_matrix > 0, axis=0)
        return dict(enumerate(item_popularity))

    def recommend_based_on_popularity(self, user_index, num_recommendations=5):
        """
        Recommend items to a user based on item popularity.

        :param user_index: Index of the user.
        :param num_recommendations: Number of recommendations to generate.
        :return: List of recommended item indices based on popularity.
        """
        item_popularity = self.get_item_popularity()
        sorted_items = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
        unrated_items = np.where(self.user_item_matrix[user_index, :] == 0)[0]

        # Recommend top popular items that are unrated by the user
        recommended_items = [item for item, _ in sorted_items if item in unrated_items]
        return recommended_items[:num_recommendations]

    def adjust_for_bias(self):
        """
        Adjust the user-item matrix for potential bias by normalizing the matrix based on global, user, and item biases.
        """
        global_mean = np.mean(self.user_item_matrix[self.user_item_matrix > 0])
        user_bias = np.mean(self.user_item_matrix, axis=1, keepdims=True) - global_mean
        item_bias = np.mean(self.user_item_matrix, axis=0, keepdims=True) - global_mean

        self.user_item_matrix = self.user_item_matrix - global_mean - user_bias - item_bias

    def recommend_with_bias_adjustment(self, user_index, num_recommendations=5):
        """
        Generate recommendations for a user with bias adjustment applied to the ratings.

        :param user_index: Index of the user.
        :param num_recommendations: Number of recommendations to generate.
        :return: List of recommended item indices after bias adjustment.
        """
        self.adjust_for_bias()
        return self.recommend(user_index, num_recommendations)

# Usage
if __name__ == "__main__":
    # Sparse matrix creation
    user_item_data = csr_matrix([
        [4, 0, 0, 5, 1],
        [5, 5, 4, 0, 0],
        [0, 0, 0, 2, 4],
        [3, 3, 0, 0, 5]
    ]).toarray()

    cf = CollaborativeFiltering(user_item_data, k=3)
    cf.fit()

    # Make predictions for a user and recommend items
    user_index = 1
    num_recommendations = 4
    recommendations = cf.recommend(user_index=user_index, num_recommendations=num_recommendations)
    print("Recommendations for user {}: {}".format(user_index, recommendations))

    # Get item popularity
    popularity = cf.get_item_popularity()
    print("Item Popularity:", popularity)

    # Recommend based on popularity
    popularity_recs = cf.recommend_based_on_popularity(user_index=user_index, num_recommendations=num_recommendations)
    print("Popularity-based recommendations for user {}: {}".format(user_index, popularity_recs))

    # Incorporate feedback
    cf.incorporate_feedback(user_index=user_index, item_index=2, feedback_rating=4)
    print("Updated recommendations after feedback for user {}: {}".format(
        user_index, cf.recommend(user_index=user_index, num_recommendations=num_recommendations)))


# Expose as a Flask microservice
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/cf_recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data['user_id']
    recommendations = cf_algorithm.recommend(user_id) 
    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)