import os
import logging
import traceback
from models.collaborative_filtering import CollaborativeFilteringModel
from models.deep_learning import DeepLearningModel
from evaluation.model_evaluation import ModelEvaluator
from utils.metrics import calculate_metrics
from data_loader import DataLoader
from configs import model_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainingPipeline")

class TrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(config['data'])
        self.models = {
            'collaborative_filtering': CollaborativeFilteringModel(config['collaborative_filtering']),
            'deep_learning': DeepLearningModel(config['deep_learning'])
        }
        self.evaluator = ModelEvaluator(config['evaluation'])
        logger.info("Initialized Training Pipeline")

        import pandas as pd
        from sklearn.preprocessing import StandardScaler, LabelEncoder

        def _preprocess_data(self, data):
            """Preprocess the data by handling missing values, normalizing numerical features, and encoding categorical variables."""
            logger.info("Preprocessing data...")

            # Handle missing values: fill missing numerical values with the median, categorical with the mode
            for column in data.select_dtypes(include=['float64', 'int64']).columns:
                data[column].fillna(data[column].median(), inplace=True)
            for column in data.select_dtypes(include=['object']).columns:
                data[column].fillna(data[column].mode()[0], inplace=True)
            logger.info("Handled missing values.")

            # Normalize numerical features using StandardScaler
            numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
            scaler = StandardScaler()
            data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
            logger.info("Normalized numerical columns.")

            # Encode categorical features using LabelEncoder
            categorical_columns = data.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                encoder = LabelEncoder()
                data[col] = encoder.fit_transform(data[col])
            logger.info("Encoded categorical columns.")

            logger.info("Data preprocessing completed.")
            return data


    def _save_model(self, model, model_name):
        """Save the trained model to a file or storage."""
        save_path = os.path.join(self.config['model_save_path'], f"{model_name}.model")
        logger.info(f"Saving model {model_name} to {save_path}")
        try:
            model.save(save_path)
            logger.info(f"Model {model_name} saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
            logger.debug(traceback.format_exc())

    def _load_previous_model(self, model_name):
        """Load the previous model if exists, for incremental training or evaluation."""
        model_path = os.path.join(self.config['model_save_path'], f"{model_name}.model")
        if os.path.exists(model_path):
            logger.info(f"Loading previously saved model {model_name} from {model_path}")
            try:
                model = self.models[model_name]
                model.load(model_path)
                logger.info(f"Loaded previous model {model_name} successfully.")
                return model
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
                logger.debug(traceback.format_exc())
        else:
            logger.info(f"No previously saved model found for {model_name}. Starting from scratch.")
        return None

    def _train_model(self, model_name, model, data):
        """Train the given model with the provided data."""
        logger.info(f"Training model: {model_name}")
        try:
            trained_model = model.train(data)
            logger.info(f"Training completed for model: {model_name}")
            return trained_model
        except Exception as e:
            logger.error(f"Training failed for model {model_name}: {e}")
            logger.debug(traceback.format_exc())
            return None

    def run(self):
        logger.info("Starting the training pipeline...")

        # Load the dataset
        try:
            data = self.data_loader.load_data()
            logger.info("Data loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            logger.debug(traceback.format_exc())
            return

        # Preprocess the data
        processed_data = self._preprocess_data(data)

        # Iterate through each model
        for model_name, model in self.models.items():
            logger.info(f"Processing model: {model_name}")

            # Load previous model if exists
            previous_model = self._load_previous_model(model_name)

            # Train the model
            if previous_model:
                logger.info(f"Using previously trained model for {model_name}.")
                trained_model = previous_model
            else:
                trained_model = self._train_model(model_name, model, processed_data)

            if trained_model:
                # Save the trained model
                self._save_model(trained_model, model_name)

                # Calculate metrics
                metrics = calculate_metrics(trained_model, processed_data)
                logger.info(f"{model_name} metrics: {metrics}")

                # Evaluate the model
                try:
                    evaluation_results = self.evaluator.evaluate(trained_model, processed_data)
                    logger.info(f"{model_name} evaluation results: {evaluation_results}")
                except Exception as e:
                    logger.error(f"Error during evaluation for {model_name}: {e}")
                    logger.debug(traceback.format_exc())
            else:
                logger.warning(f"Skipping evaluation and saving for {model_name} due to training failure.")

        logger.info("Training pipeline completed.")

    def validate_config(self):
        """Validate the configuration settings before running the pipeline."""
        logger.info("Validating configuration settings...")
        required_keys = ['data', 'model_save_path', 'evaluation', 'collaborative_filtering', 'deep_learning']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        logger.info("Configuration validation completed successfully.")

    def cleanup(self):
        """Cleanup resources after training completes."""
        logger.info("Cleaning up resources...")
        # Closing connections, releasing file handles, etc
        logger.info("Cleanup completed.")

        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        def notify_completion(self):
            """Send an email notification that the training pipeline has completed successfully."""
            logger.info("Notifying completion of the training pipeline...")

            # Email configuration
            sender_email = "person@website.com"
            receiver_email = "person2@website.com"
            password = "password"

            # Create email content
            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = receiver_email
            message["Subject"] = "Training Pipeline Completion Notification"

            body = "Hello,\n\nThe training pipeline has successfully completed.\n\nBest regards,\nTraining Pipeline Team"
            message.attach(MIMEText(body, "plain"))

            try:
                # Set up the SMTP server and send the email
                with smtplib.SMTP("smtp.website.com", 587) as server:
                    server.starttls()
                    server.login(sender_email, password)
                    server.sendmail(sender_email, receiver_email, message.as_string())
                logger.info("Completion notification email sent successfully.")
            except Exception as e:
                logger.error(f"Failed to send completion notification: {e}")
                logger.debug(traceback.format_exc())


if __name__ == "__main__":
    try:
        logger.info("Loading configuration...")
        pipeline = TrainingPipeline(model_config)

        pipeline.validate_config()  # Ensure the config is valid before running the pipeline
        pipeline.run()  # Run the training pipeline

        pipeline.notify_completion()  # Notify once training is complete
    except Exception as e:
        logger.error(f"Pipeline encountered an error: {e}")
        logger.debug(traceback.format_exc())
    finally:
        pipeline.cleanup()