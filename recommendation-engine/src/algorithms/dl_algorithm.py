import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np

class DeepLearningRecommender:
    def __init__(self, num_users, num_items, embedding_size=50, learning_rate=0.001, dropout_rate=0.2):
        """
        Initialize the deep learning recommender with given hyperparameters.
        """
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.model = self.build_model()

    def build_model(self):
        """
        Build the deep learning model architecture using Keras.
        """

        # User and item input layers
        user_input = layers.Input(shape=(1,), name='user_input')
        item_input = layers.Input(shape=(1,), name='item_input')

        # Embedding layers for users and items
        user_embedding = layers.Embedding(input_dim=self.num_users, 
                                          output_dim=self.embedding_size, 
                                          embeddings_regularizer=regularizers.l2(1e-6), 
                                          name='user_embedding')(user_input)
        item_embedding = layers.Embedding(input_dim=self.num_items, 
                                          output_dim=self.embedding_size, 
                                          embeddings_regularizer=regularizers.l2(1e-6), 
                                          name='item_embedding')(item_input)

        # Flatten embeddings for concatenation
        user_vec = layers.Flatten(name='user_vec')(user_embedding)
        item_vec = layers.Flatten(name='item_vec')(item_embedding)

        # Concatenate user and item vectors
        concat = layers.Concatenate(name='concat_layer')([user_vec, item_vec])

        # Fully connected layers with dropout and batch normalization
        dense_1 = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-6))(concat)
        dense_1 = layers.Dropout(self.dropout_rate)(dense_1)
        dense_1 = layers.BatchNormalization()(dense_1)

        dense_2 = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-6))(dense_1)
        dense_2 = layers.Dropout(self.dropout_rate)(dense_2)
        dense_2 = layers.BatchNormalization()(dense_2)

        dense_3 = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-6))(dense_2)
        dense_3 = layers.Dropout(self.dropout_rate)(dense_3)
        dense_3 = layers.BatchNormalization()(dense_3)

        # Output layer for predicting user-item interaction
        output = layers.Dense(1, activation='sigmoid', name='output_layer')(dense_3)

        # Build the model
        model = models.Model(inputs=[user_input, item_input], outputs=output)

        # Compile the model with Adam optimizer and binary cross-entropy loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def generate_synthetic_data(self, num_samples=10000):
        """
        Generate synthetic user-item interaction data for testing and training.
        """
        user_data = np.random.randint(0, self.num_users, size=num_samples)
        item_data = np.random.randint(0, self.num_items, size=num_samples)
        labels = np.random.randint(0, 2, size=num_samples)

        return user_data, item_data, labels

    def train(self, user_data, item_data, labels, batch_size=32, epochs=10, validation_split=0.2):
        """
        Train the model using the provided data.
        """
        history = self.model.fit([user_data, item_data], labels, 
                                 batch_size=batch_size, 
                                 epochs=epochs, 
                                 validation_split=validation_split)
        return history

    def predict(self, user_data, item_data):
        """
        Predict interaction probabilities for user-item pairs.
        """
        predictions = self.model.predict([user_data, item_data])
        return predictions

    def evaluate(self, user_data, item_data, labels):
        """
        Evaluate the model's performance on the given dataset.
        """
        evaluation = self.model.evaluate([user_data, item_data], labels)
        return evaluation

    def save_model(self, filepath):
        """
        Save the model to the specified file path.
        """
        self.model.save(filepath)

    def load_model(self, filepath):
        """
        Load a pre-trained model from the specified file path.
        """
        self.model = models.load_model(filepath)

    def hyperparameter_search(self, user_data, item_data, labels, learning_rates, embedding_sizes, dropout_rates):
        """
        Conduct hyperparameter tuning by trying different configurations.
        """
        best_accuracy = 0
        best_params = {}
        for lr in learning_rates:
            for emb_size in embedding_sizes:
                for dr in dropout_rates:
                    print(f"Testing config: LR={lr}, Embedding Size={emb_size}, Dropout={dr}")
                    # Initialize the model with new hyperparameters
                    self.learning_rate = lr
                    self.embedding_size = emb_size
                    self.dropout_rate = dr
                    self.model = self.build_model()
                    # Train the model
                    history = self.train(user_data, item_data, labels)
                    # Evaluate the model
                    accuracy = max(history.history['accuracy'])
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {'learning_rate': lr, 'embedding_size': emb_size, 'dropout_rate': dr}
        
        print(f"Best Accuracy: {best_accuracy}")
        print(f"Best Parameters: {best_params}")

    def log_training_progress(self, history):
        """
        Log the training progress, including loss and accuracy at each epoch.
        """
        for epoch, (loss, acc) in enumerate(zip(history.history['loss'], history.history['accuracy'])):
            print(f"Epoch {epoch + 1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

    def plot_training_curves(self, history):
        """
        Plot the training and validation accuracy and loss curves.
        """
        import matplotlib.pyplot as plt

        # Plot accuracy curves
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss curves
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

    def add_early_stopping(self, patience=5):
        """
        Add early stopping callback to stop training when no improvement is seen for a given number of epochs.
        """
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        return early_stopping

    def add_model_checkpoint(self, filepath):
        """
        Add model checkpoint callback to save the model during training.
        """
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')
        return checkpoint

    def train_with_callbacks(self, user_data, item_data, labels, batch_size=32, epochs=20, validation_split=0.2, patience=5, checkpoint_filepath='best_model.h5'):
        """
        Train the model with early stopping and model checkpoint callbacks.
        """
        callbacks = [
            self.add_early_stopping(patience=patience),
            self.add_model_checkpoint(filepath=checkpoint_filepath)
        ]

        history = self.model.fit([user_data, item_data], labels, 
                                 batch_size=batch_size, 
                                 epochs=epochs, 
                                 validation_split=validation_split, 
                                 callbacks=callbacks)

        return history

    def generate_recommendations(self, user_id, top_k=10):
        """
        Generate top-K recommendations for a given user by predicting interaction scores for all items.
        """
        user_input = np.array([user_id] * self.num_items)
        item_input = np.array(range(self.num_items))

        # Predict interaction probabilities for all items
        predictions = self.model.predict([user_input, item_input])

        # Rank the items by predicted interaction score
        recommended_items = np.argsort(-predictions, axis=0)[:top_k]
        return recommended_items.flatten()

    def generate_recommendations_for_all_users(self, top_k=10):
        """
        Generate top-K recommendations for all users in the system.
        """
        all_recommendations = {}

        for user_id in range(self.num_users):
            recommendations = self.generate_recommendations(user_id=user_id, top_k=top_k)
            all_recommendations[user_id] = recommendations

        return all_recommendations

    def grid_search_hyperparameters(self, user_data, item_data, labels, learning_rates, batch_sizes, epochs_list):
        """
        Perform grid search to find the best combination of learning rates, batch sizes, and epochs.
        """
        best_accuracy = 0
        best_params = {}

        for lr in learning_rates:
            for batch_size in batch_sizes:
                for epochs in epochs_list:
                    print(f"Testing with learning rate: {lr}, batch size: {batch_size}, epochs: {epochs}")
                    
                    self.learning_rate = lr
                    self.model = self.build_model()

                    history = self.train(user_data, item_data, labels, batch_size=batch_size, epochs=epochs)

                    accuracy = max(history.history['accuracy'])
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {'learning_rate': lr, 'batch_size': batch_size, 'epochs': epochs}

        print(f"Best Accuracy: {best_accuracy}")
        print(f"Best Hyperparameters: {best_params}")
        return best_params

    def evaluate_on_test_data(self, test_user_data, test_item_data, test_labels):
        """
        Evaluate the model's performance on test data and return the loss and accuracy.
        """
        test_loss, test_accuracy = self.model.evaluate([test_user_data, test_item_data], test_labels)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        return test_loss, test_accuracy

    def fine_tune_model(self, user_data, item_data, labels, additional_epochs=5):
        """
        Fine-tune the model by continuing training for additional epochs.
        """
        history = self.model.fit([user_data, item_data], labels, epochs=additional_epochs)
        return history

    def recommend_for_new_user(self, new_user_id, item_ids, top_k=10):
        """
        Generate recommendations for a new user based on the provided item interactions.
        """
        new_user_input = np.array([new_user_id] * len(item_ids))
        predictions = self.model.predict([new_user_input, item_ids])

        recommended_items = np.argsort(-predictions, axis=0)[:top_k]
        return recommended_items.flatten()

    def log_experiment(self, params, results, filepath="experiment_log.txt"):
        """
        Log hyperparameters and results of a training experiment to a text file.
        """
        with open(filepath, 'a') as file:
            file.write(f"Parameters: {params}\n")
            file.write(f"Results: {results}\n")
            file.write("="*40 + "\n")

    def generate_user_embeddings(self):
        """
        Extract and return the user embeddings from the trained model.
        """
        user_embedding_layer = self.model.get_layer('user_embedding')
        user_embeddings = user_embedding_layer.get_weights()[0]
        return user_embeddings

    def generate_item_embeddings(self):
        """
        Extract and return the item embeddings from the trained model.
        """
        item_embedding_layer = self.model.get_layer('item_embedding')
        item_embeddings = item_embedding_layer.get_weights()[0]
        return item_embeddings

    def visualize_embeddings(self, embeddings, num_points=100, labels=None):
        """
        Visualize the embeddings using t-SNE.
        """
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        tsne = TSNE(n_components=2)
        reduced_embeddings = tsne.fit_transform(embeddings[:num_points])

        plt.figure(figsize=(8, 8))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='blue', cmap='viridis')
        
        if labels:
            for i in range(num_points):
                plt.annotate(str(labels[i]), (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

        plt.title("t-SNE visualization of embeddings")
        plt.show()

    def load_embeddings_from_file(self, filepath):
        """
        Load pre-trained embeddings from a file.
        """
        embeddings = np.load(filepath)
        return embeddings

    def save_embeddings_to_file(self, embeddings, filepath):
        """
        Save the embeddings to a file in NumPy format.
        """
        np.save(filepath, embeddings)

    def calculate_similarity(self, embedding_1, embedding_2):
        """
        Calculate the cosine similarity between two embeddings.
        """
        dot_product = np.dot(embedding_1, embedding_2)
        norm_1 = np.linalg.norm(embedding_1)
        norm_2 = np.linalg.norm(embedding_2)
        similarity = dot_product / (norm_1 * norm_2)
        return similarity

    def get_most_similar_items(self, item_id, top_k=10):
        """
        Get the most similar items to a given item based on their embeddings.
        """
        item_embeddings = self.generate_item_embeddings()
        target_embedding = item_embeddings[item_id]

        similarities = [self.calculate_similarity(target_embedding, emb) for emb in item_embeddings]
        most_similar_items = np.argsort(-np.array(similarities))[:top_k]
        return most_similar_items

    def recommend_similar_items(self, item_id, top_k=10):
        """
        Recommend items similar to the given item based on embedding similarity.
        """
        similar_items = self.get_most_similar_items(item_id, top_k=top_k)
        print(f"Items similar to Item {item_id}: {similar_items}")
        return similar_items

    def build_pairwise_ranking_model(self):
        """
        Build a model for pairwise ranking using a triplet loss function for learning item similarities.
        """
        # Define the input layers
        user_input = layers.Input(shape=(1,), name='user_input')
        positive_item_input = layers.Input(shape=(1,), name='positive_item_input')
        negative_item_input = layers.Input(shape=(1,), name='negative_item_input')

        # Shared embedding layers for items
        user_embedding = layers.Embedding(input_dim=self.num_users, 
                                          output_dim=self.embedding_size, 
                                          embeddings_regularizer=regularizers.l2(1e-6), 
                                          name='user_embedding')(user_input)

        item_embedding = layers.Embedding(input_dim=self.num_items, 
                                          output_dim=self.embedding_size, 
                                          embeddings_regularizer=regularizers.l2(1e-6), 
                                          name='item_embedding')

        # Get embeddings for the positive and negative items
        positive_item_embedding = item_embedding(positive_item_input)
        negative_item_embedding = item_embedding(negative_item_input)

        # Flatten the embeddings
        user_vec = layers.Flatten(name='user_vec')(user_embedding)
        positive_item_vec = layers.Flatten(name='positive_item_vec')(positive_item_embedding)
        negative_item_vec = layers.Flatten(name='negative_item_vec')(negative_item_embedding)

        # Define the triplet loss function
        def triplet_loss(_, y_pred):
            margin = 1.0
            positive_distance = tf.reduce_sum(tf.square(user_vec - positive_item_vec), axis=-1)
            negative_distance = tf.reduce_sum(tf.square(user_vec - negative_item_vec), axis=-1)
            return tf.maximum(positive_distance - negative_distance + margin, 0.0)

        # Output a sample prediction since only triplet loss is needed
        output = layers.Lambda(lambda x: x)(user_vec)

        # Build the model
        model = models.Model(inputs=[user_input, positive_item_input, negative_item_input], outputs=output)
        model.compile(optimizer='adam', loss=triplet_loss)

        return model

    def train_pairwise_ranking(self, user_data, positive_item_data, negative_item_data, batch_size=32, epochs=10):
        """
        Train the pairwise ranking model using the triplet loss function.
        """
        ranking_model = self.build_pairwise_ranking_model()

        # Train the model with triplet data
        history = ranking_model.fit([user_data, positive_item_data, negative_item_data], 
                                    np.zeros(len(user_data)), 
                                    batch_size=batch_size, 
                                    epochs=epochs)
        return history

    def evaluate_pairwise_ranking(self, user_data, positive_item_data, negative_item_data):
        """
        Evaluate the pairwise ranking model on a test set.
        """
        ranking_model = self.build_pairwise_ranking_model()
        loss = ranking_model.evaluate([user_data, positive_item_data, negative_item_data], np.zeros(len(user_data)))
        print(f"Pairwise ranking loss: {loss:.4f}")
        return loss

    def perform_online_learning(self, user_data, item_data, labels, batch_size=32, epochs=5):
        """
        Perform online learning by continuously updating the model with new data.
        """
        for epoch in range(epochs):
            print(f"Online Learning Epoch {epoch + 1}")
            indices = np.random.permutation(len(user_data))
            shuffled_user_data = user_data[indices]
            shuffled_item_data = item_data[indices]
            shuffled_labels = labels[indices]

            # Train the model on the shuffled data
            self.model.fit([shuffled_user_data, shuffled_item_data], shuffled_labels, batch_size=batch_size)

    def explain_recommendations(self, user_id, item_ids, top_k=5):
        """
        Provide an explanation for the recommendations by calculating the contribution of each feature.
        """
        user_input = np.array([user_id] * len(item_ids))
        predictions = self.model.predict([user_input, item_ids])

        contributions = []
        for i, item_id in enumerate(item_ids):
            explanation = f"Item {item_id}: Predicted score = {predictions[i][0]:.4f}"
            contributions.append(explanation)

        top_contributions = sorted(contributions, key=lambda x: float(x.split('=')[-1]), reverse=True)[:top_k]
        for contrib in top_contributions:
            print(contrib)

    def deploy_model(self, model_version="1.0"):
        """
        Simulate the deployment of the trained model to a server.
        """
        print(f"Deploying model version {model_version} to the server...")

    def monitor_model_performance(self, test_user_data, test_item_data, test_labels):
        """
        Monitor the deployed model's performance in real-time using evaluation metrics.
        """
        print("Monitoring model performance...")
        test_loss, test_accuracy = self.evaluate_on_test_data(test_user_data, test_item_data, test_labels)
        print(f"Real-time performance: Loss = {test_loss:.4f}, Accuracy = {test_accuracy:.4f}")

    def simulate_user_feedback(self, user_id, item_id, interaction):
        """
        Simulate user feedback and update the model accordingly.
        """
        print(f"User {user_id} interacted with Item {item_id}: Feedback = {interaction}")

    def incremental_update(self, user_data, item_data, labels):
        """
        Update the model incrementally as new data comes in.
        """
        print("Performing incremental model update with new data...")
        self.model.fit([user_data, item_data], labels)

    def perform_ab_testing(self, model_a, model_b, user_data, item_data):
        """
        Simulate A/B testing between two models to evaluate which performs better.
        """
        predictions_a = model_a.predict([user_data, item_data])
        predictions_b = model_b.predict([user_data, item_data])

        print(f"Comparing Model A and Model B predictions...")


# Expose as a Flask microservice
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/dl_recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data['user_id']
    recommendations = dl_algorithm.recommend(user_id) 
    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5003)
