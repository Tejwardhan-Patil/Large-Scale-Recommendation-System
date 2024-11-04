import numpy as np
from scipy.sparse import csr_matrix

class CollaborativeFiltering:
    def __init__(self, num_users, num_items, latent_features=10, alpha=0.01, beta=0.01, iterations=1000):
        """
        Initialize the collaborative filtering model with the given parameters.
        
        :param num_users: Total number of users in the system
        :param num_items: Total number of items in the system
        :param latent_features: Number of latent features for matrix factorization
        :param alpha: Learning rate for gradient descent
        :param beta: Regularization parameter
        :param iterations: Number of iterations for training
        """
        self.num_users = num_users
        self.num_items = num_items
        self.latent_features = latent_features
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

        # Initialize the user and item latent feature matrices
        self.user_features = np.random.normal(scale=1.0 / self.latent_features, 
                                              size=(self.num_users, self.latent_features))
        self.item_features = np.random.normal(scale=1.0 / self.latent_features, 
                                              size=(self.num_items, self.latent_features))

    def train(self, ratings):
        """
        Train the model using the provided ratings matrix.
        
        :param ratings: List of tuples (user, item, rating)
        """
        for iteration in range(self.iterations):
            # Perform SGD for each user-item-rating tuple in the training data
            for user, item, rating in ratings:
                # Predict the current rating
                prediction = self.predict(user, item)
                
                # Calculate the error
                error = rating - prediction
                
                # Update user and item latent feature matrices using gradient descent
                self.user_features[user, :] += self.alpha * (error * self.item_features[item, :] - self.beta * self.user_features[user, :])
                self.item_features[item, :] += self.alpha * (error * self.user_features[user, :] - self.beta * self.item_features[item, :])
                
            # Compute the loss and print it every 10 iterations
            if iteration % 10 == 0:
                loss = self.compute_loss(ratings)
                print(f"Iteration: {iteration}, Loss: {loss}")

    def predict(self, user, item):
        """
        Predict the rating for a given user-item pair.
        
        :param user: User index
        :param item: Item index
        :return: Predicted rating for the user-item pair
        """
        return np.dot(self.user_features[user, :], self.item_features[item, :])

    def compute_loss(self, ratings):
        """
        Compute the loss for the given ratings matrix using Mean Squared Error (MSE).
        
        :param ratings: List of tuples (user, item, rating)
        :return: Computed loss value
        """
        loss = 0
        for user, item, rating in ratings:
            prediction = self.predict(user, item)
            loss += (rating - prediction) ** 2
        return loss

    def recommend(self, user, num_recommendations=5):
        """
        Recommend top N items for a given user based on their latent features.
        
        :param user: User index for which recommendations are generated
        :param num_recommendations: Number of recommendations to generate
        :return: List of item indices representing the top recommendations
        """
        # Predict ratings for all items for the given user
        predictions = np.dot(self.user_features[user, :], self.item_features.T)
        
        # Sort the predictions in descending order and return the top N item indices
        recommendations = np.argsort(predictions)[-num_recommendations:][::-1]
        return recommendations

    def get_user_features(self):
        """
        Get the user latent feature matrix.
        
        :return: User feature matrix
        """
        return self.user_features

    def get_item_features(self):
        """
        Get the item latent feature matrix.
        
        :return: Item feature matrix
        """
        return self.item_features

    def save_model(self, user_feature_file, item_feature_file):
        """
        Save the trained user and item latent features to files.
        
        :param user_feature_file: File to save user latent features
        :param item_feature_file: File to save item latent features
        """
        np.save(user_feature_file, self.user_features)
        np.save(item_feature_file, self.item_features)

    def load_model(self, user_feature_file, item_feature_file):
        """
        Load the user and item latent features from files.
        
        :param user_feature_file: File to load user latent features from
        :param item_feature_file: File to load item latent features from
        """
        self.user_features = np.load(user_feature_file)
        self.item_features = np.load(item_feature_file)

    def update_user_features(self, user, new_features):
        """
        Update the latent features for a specific user.
        
        :param user: User index
        :param new_features: New feature vector for the user
        """
        if len(new_features) != self.latent_features:
            raise ValueError(f"Feature vector length must be {self.latent_features}")
        self.user_features[user, :] = new_features

    def update_item_features(self, item, new_features):
        """
        Update the latent features for a specific item.
        
        :param item: Item index
        :param new_features: New feature vector for the item
        """
        if len(new_features) != self.latent_features:
            raise ValueError(f"Feature vector length must be {self.latent_features}")
        self.item_features[item, :] = new_features

    def add_user(self, new_user_features=None):
        """
        Add a new user to the system, initializing their latent features.
        
        :param new_user_features: Parameter for new user features
        :return: Index of the newly added user
        """
        if new_user_features is None:
            new_user_features = np.random.normal(scale=1.0 / self.latent_features, size=(1, self.latent_features))
        else:
            if len(new_user_features) != self.latent_features:
                raise ValueError(f"Feature vector length must be {self.latent_features}")
        self.user_features = np.vstack([self.user_features, new_user_features])
        self.num_users += 1
        return self.num_users - 1

    def add_item(self, new_item_features=None):
        """
        Add a new item to the system, initializing its latent features.
        
        :param new_item_features: Parameter for new item features
        :return: Index of the newly added item
        """
        if new_item_features is None:
            new_item_features = np.random.normal(scale=1.0 / self.latent_features, size=(1, self.latent_features))
        else:
            if len(new_item_features) != self.latent_features:
                raise ValueError(f"Feature vector length must be {self.latent_features}")
        self.item_features = np.vstack([self.item_features, new_item_features])
        self.num_items += 1
        return self.num_items - 1

    def remove_user(self, user):
        """
        Remove a user from the system by their index.
        
        :param user: User index to remove
        """
        if user < 0 or user >= self.num_users:
            raise IndexError(f"User index {user} is out of bounds")
        self.user_features = np.delete(self.user_features, user, axis=0)
        self.num_users -= 1

    def remove_item(self, item):
        """
        Remove an item from the system by its index.
        
        :param item: Item index to remove
        """
        if item < 0 or item >= self.num_items:
            raise IndexError(f"Item index {item} is out of bounds")
        self.item_features = np.delete(self.item_features, item, axis=0)
        self.num_items -= 1

    def normalize_ratings(self, ratings):
        """
        Normalize the ratings by subtracting the mean rating for each user.
        
        :param ratings: List of tuples (user, item, rating)
        :return: Normalized ratings, user biases
        """
        user_ratings = {}
        for user, item, rating in ratings:
            if user not in user_ratings:
                user_ratings[user] = []
            user_ratings[user].append(rating)
        
        user_biases = {}
        normalized_ratings = []
        for user, ratings_list in user_ratings.items():
            user_biases[user] = np.mean(ratings_list)
            for item, rating in zip([r[1] for r in ratings if r[0] == user], ratings_list):
                normalized_ratings.append((user, item, rating - user_biases[user]))
        return normalized_ratings, user_biases

    def denormalize_predictions(self, predictions, user_biases):
        """
        Denormalize the predicted ratings by adding the user bias.
        
        :param predictions: Dictionary of predicted ratings
        :param user_biases: Dictionary of user biases
        :return: Denormalized predictions
        """
        denormalized_predictions = {}
        for user, items in predictions.items():
            denormalized_predictions[user] = {item: rating + user_biases[user] for item, rating in items.items()}
        return denormalized_predictions

    def top_n_recommendations(self, ratings, n=5):
        """
        Generate top N recommendations for all users.
        
        :param ratings: List of tuples (user, item, rating)
        :param n: Number of recommendations to generate per user
        :return: Dictionary of users and their top N recommended items
        """
        user_item_ratings = {}
        for user, item, rating in ratings:
            if user not in user_item_ratings:
                user_item_ratings[user] = {}
            user_item_ratings[user][item] = rating
        
        recommendations = {}
        for user in range(self.num_users):
            user_rated_items = user_item_ratings.get(user, {})
            predicted_ratings = self._predict_for_user(user, user_rated_items.keys())
            sorted_items = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)
            recommendations[user] = [item for item, _ in sorted_items[:n]]
        
        return recommendations

    def _predict_for_user(self, user, rated_items):
        """
        Predict ratings for all items that the user hasn't rated.
        
        :param user: User index
        :param rated_items: Set of item indices that the user has already rated
        :return: Predicted ratings for the unrated items
        """
        predictions = {}
        for item in range(self.num_items):
            if item not in rated_items:
                predictions[item] = self.predict(user, item)
        return predictions

    def batch_train(self, ratings, batch_size=100):
        """
        Train the model using mini-batch gradient descent.
        
        :param ratings: List of tuples (user, item, rating)
        :param batch_size: Size of each training batch
        """
        np.random.shuffle(ratings)
        batches = [ratings[k:k + batch_size] for k in range(0, len(ratings), batch_size)]
        
        for batch in batches:
            user_updates = np.zeros_like(self.user_features)
            item_updates = np.zeros_like(self.item_features)
            
            for user, item, rating in batch:
                prediction = self.predict(user, item)
                error = rating - prediction
                user_updates[user, :] += self.alpha * (error * self.item_features[item, :] - self.beta * self.user_features[user, :])
                item_updates[item, :] += self.alpha * (error * self.user_features[user, :] - self.beta * self.item_features[item, :])
            
            self.user_features += user_updates
            self.item_features += item_updates

    def cross_validate(self, ratings, k_folds=5):
        """
        Perform k-fold cross-validation to evaluate the model performance.
        
        :param ratings: List of tuples (user, item, rating)
        :param k_folds: Number of cross-validation folds
        :return: Average loss across all folds
        """
        np.random.shuffle(ratings)
        fold_size = len(ratings) // k_folds
        losses = []
        
        for fold in range(k_folds):
            test_set = ratings[fold * fold_size:(fold + 1) * fold_size]
            train_set = ratings[:fold * fold_size] + ratings[(fold + 1) * fold_size:]
            
            # Train on the training set
            self.train(train_set)
            
            # Evaluate on the test set
            loss = self.compute_loss(test_set)
            losses.append(loss)
            print(f"Fold {fold + 1}, Loss: {loss}")
        
        return np.mean(losses)

    def evaluate_model(self, ratings):
        """
        Evaluate the model using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
        
        :param ratings: List of tuples (user, item, rating)
        :return: Tuple of (MAE, RMSE)
        """
        mae = 0
        mse = 0
        for user, item, rating in ratings:
            prediction = self.predict(user, item)
            error = abs(rating - prediction)
            mae += error
            mse += error ** 2
        
        mae /= len(ratings)
        rmse = np.sqrt(mse / len(ratings))
        
        return mae, rmse

    def generate_synthetic_data(self, num_samples=1000):
        """
        Generate synthetic user-item-rating data for testing purposes.
        
        :param num_samples: Number of synthetic samples to generate
        :return: List of tuples (user, item, rating)
        """
        ratings = []
        for _ in range(num_samples):
            user = np.random.randint(0, self.num_users)
            item = np.random.randint(0, self.num_items)
            rating = np.random.uniform(1, 5)
            ratings.append((user, item, rating))
        
        return ratings

    def save_recommendations(self, recommendations, file_path):
        """
        Save the generated recommendations to a file.
        
        :param recommendations: Dictionary of user recommendations
        :param file_path: Path to save the recommendations file
        """
        with open(file_path, 'w') as file:
            for user, items in recommendations.items():
                file.write(f"User {user}: {', '.join(map(str, items))}\n")

    def load_ratings(self, file_path):
        """
        Load user-item-rating data from a file.
        
        :param file_path: Path to the ratings file
        :return: List of tuples (user, item, rating)
        """
        ratings = []
        with open(file_path, 'r') as file:
            for line in file:
                user, item, rating = line.strip().split(',')
                ratings.append((int(user), int(item), float(rating)))
        return ratings

    def print_recommendations(self, recommendations):
        """
        Print the generated recommendations to the console.
        
        :param recommendations: Dictionary of user recommendations
        """
        for user, items in recommendations.items():
            print(f"User {user}: {', '.join(map(str, items))}")

    def set_learning_rate(self, new_alpha):
        """
        Update the learning rate for gradient descent.
        
        :param new_alpha: New learning rate
        """
        if new_alpha <= 0:
            raise ValueError("Learning rate must be positive")
        self.alpha = new_alpha

    def set_regularization(self, new_beta):
        """
        Update the regularization parameter.
        
        :param new_beta: New regularization value
        """
        if new_beta < 0:
            raise ValueError("Regularization parameter must be non-negative")
        self.beta = new_beta

    def optimize(self, ratings, learning_rate_schedule=None, regularization_schedule=None):
        """
        Optimize the model parameters using dynamic learning rate and regularization schedules.
        
        :param ratings: List of tuples (user, item, rating)
        :param learning_rate_schedule: List of tuples (iteration, learning_rate)
        :param regularization_schedule: List of tuples (iteration, regularization_value)
        """
        for iteration in range(self.iterations):
            if learning_rate_schedule:
                for step, lr in learning_rate_schedule:
                    if iteration == step:
                        self.alpha = lr
                        print(f"Iteration {iteration}: Updated learning rate to {lr}")

            if regularization_schedule:
                for step, reg in regularization_schedule:
                    if iteration == step:
                        self.beta = reg
                        print(f"Iteration {iteration}: Updated regularization to {reg}")

            for user, item, rating in ratings:
                prediction = self.predict(user, item)
                error = rating - prediction
                self.user_features[user, :] += self.alpha * (error * self.item_features[item, :] - self.beta * self.user_features[user, :])
                self.item_features[item, :] += self.alpha * (error * self.user_features[user, :] - self.beta * self.item_features[item, :])

            if iteration % 10 == 0:
                loss = self.compute_loss(ratings)
                print(f"Iteration: {iteration}, Loss: {loss}")

    def grid_search(self, ratings, param_grid):
        """
        Perform a grid search over hyperparameters to find the best combination.
        
        :param ratings: List of tuples (user, item, rating)
        :param param_grid: Dictionary of hyperparameters with lists of possible values
        :return: Best hyperparameter combination
        """
        best_params = None
        best_loss = float('inf')

        for latent_features in param_grid.get('latent_features', [self.latent_features]):
            for alpha in param_grid.get('alpha', [self.alpha]):
                for beta in param_grid.get('beta', [self.beta]):
                    print(f"Testing params: latent_features={latent_features}, alpha={alpha}, beta={beta}")
                    self.latent_features = latent_features
                    self.alpha = alpha
                    self.beta = beta

                    # Reinitialize feature matrices
                    self.user_features = np.random.normal(scale=1.0/self.latent_features, size=(self.num_users, self.latent_features))
                    self.item_features = np.random.normal(scale=1.0/self.latent_features, size=(self.num_items, self.latent_features))

                    # Train and evaluate model
                    self.train(ratings)
                    loss = self.compute_loss(ratings)
                    print(f"Loss: {loss}")
                    
                    if loss < best_loss:
                        best_loss = loss
                        best_params = {'latent_features': latent_features, 'alpha': alpha, 'beta': beta}

        print(f"Best params: {best_params}, Best loss: {best_loss}")
        return best_params

    def adaptive_training(self, ratings, loss_threshold=0.001, patience=5):
        """
        Train the model adaptively, stopping early if the improvement is below a certain threshold.
        
        :param ratings: List of tuples (user, item, rating)
        :param loss_threshold: Minimum loss improvement required to continue training
        :param patience: Number of iterations to wait before early stopping
        """
        previous_loss = None
        wait = 0
        
        for iteration in range(self.iterations):
            for user, item, rating in ratings:
                prediction = self.predict(user, item)
                error = rating - prediction
                self.user_features[user, :] += self.alpha * (error * self.item_features[item, :] - self.beta * self.user_features[user, :])
                self.item_features[item, :] += self.alpha * (error * self.user_features[user, :] - self.beta * self.item_features[item, :])
            
            current_loss = self.compute_loss(ratings)
            print(f"Iteration: {iteration}, Loss: {current_loss}")
            
            if previous_loss is not None and abs(previous_loss - current_loss) < loss_threshold:
                wait += 1
                if wait >= patience:
                    print("Early stopping triggered.")
                    break
            else:
                wait = 0
            
            previous_loss = current_loss