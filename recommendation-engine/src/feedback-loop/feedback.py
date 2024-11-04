import json
import logging
from typing import List, Dict, Any, Optional
from statistics import mean, stdev

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeedbackProcessor:
    def __init__(self, feedback_store: str, user_data_store: str):
        """
        Initializes the feedback processor with paths to feedback and user data stores.
        """
        self.feedback_store = feedback_store
        self.user_data_store = user_data_store

    def load_feedback(self) -> List[Dict[str, Any]]:
        """
        Loads feedback data from a JSON file.
        Returns a list of feedback dictionaries.
        """
        logging.info(f"Loading feedback data from {self.feedback_store}")
        try:
            with open(self.feedback_store, 'r') as file:
                feedback_data = json.load(file)
            logging.info("Feedback data loaded successfully")
            return feedback_data
        except FileNotFoundError:
            logging.error(f"File {self.feedback_store} not found")
            return []
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from {self.feedback_store}")
            return []

    def validate_feedback(self, feedback: Dict[str, Any]) -> bool:
        """
        Validates the feedback data. Ensures each feedback entry contains required fields.
        """
        required_fields = {"user_id", "item_id", "rating"}
        if not all(field in feedback for field in required_fields):
            logging.warning(f"Invalid feedback entry: {feedback}")
            return False
        if not isinstance(feedback['rating'], (int, float)):
            logging.warning(f"Invalid rating type in feedback: {feedback}")
            return False
        return True

    def load_user_data(self) -> Dict[str, Any]:
        """
        Loads user data for user-specific operations.
        Returns a dictionary with user-specific information.
        """
        logging.info(f"Loading user data from {self.user_data_store}")
        try:
            with open(self.user_data_store, 'r') as file:
                user_data = json.load(file)
            logging.info("User data loaded successfully")
            return user_data
        except FileNotFoundError:
            logging.error(f"File {self.user_data_store} not found")
            return {}
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from {self.user_data_store}")
            return {}

    def process_feedback(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """
        Processes feedback data and summarizes ratings by user.
        Returns a dictionary mapping user IDs to a list of ratings.
        """
        logging.info("Processing feedback data")
        feedback_summary = {}
        for feedback in feedback_data:
            if self.validate_feedback(feedback):
                user_id = feedback['user_id']
                rating = feedback['rating']
                if user_id not in feedback_summary:
                    feedback_summary[user_id] = []
                feedback_summary[user_id].append(rating)
        logging.info("Feedback processing complete")
        return feedback_summary

    def calculate_statistics(self, ratings: List[float]) -> Dict[str, Optional[float]]:
        """
        Calculates basic statistics for a list of ratings, including mean and standard deviation.
        Returns a dictionary with statistical information.
        """
        if not ratings:
            return {"mean": None, "std_dev": None}
        return {
            "mean": mean(ratings),
            "std_dev": stdev(ratings) if len(ratings) > 1 else 0.0
        }

    def update_model(self, feedback_summary: Dict[str, List[float]], user_data: Dict[str, Any]):
        """
        Updates the recommendation model based on the feedback summary.
        """
        logging.info("Updating model with feedback data")
        for user_id, ratings in feedback_summary.items():
            stats = self.calculate_statistics(ratings)
            avg_rating = stats['mean']
            if avg_rating is not None:
                self._update_user_preferences(user_id, avg_rating, user_data)

    def _update_user_preferences(self, user_id: str, avg_rating: float, user_data: Dict[str, Any]):
        """
        Updates user preferences in the recommendation model based on the average rating.
        """
        logging.info(f"Updating preferences for user {user_id} with average rating {avg_rating}")
        if user_id in user_data:
            user_preferences = user_data.get(user_id, {}).get("preferences", {})
            user_preferences['average_rating'] = avg_rating
            logging.info(f"User {user_id} preferences updated to {user_preferences}")
        else:
            logging.warning(f"User {user_id} not found in user data")

    def save_user_data(self, user_data: Dict[str, Any]):
        """
        Saves updated user data back to the user data store.
        """
        logging.info(f"Saving updated user data to {self.user_data_store}")
        try:
            with open(self.user_data_store, 'w') as file:
                json.dump(user_data, file, indent=4)
            logging.info("User data saved successfully")
        except Exception as e:
            logging.error(f"Failed to save user data: {e}")

    def generate_feedback_report(self, feedback_summary: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Generates a report summarizing the feedback.
        """
        report = {
            "total_users": len(feedback_summary),
            "user_stats": {}
        }
        for user_id, ratings in feedback_summary.items():
            report["user_stats"][user_id] = self.calculate_statistics(ratings)
        logging.info("Feedback report generated")
        return report

    def display_feedback_report(self, report: Dict[str, Any]):
        """
        Displays the feedback report in a user-friendly format.
        """
        logging.info("Displaying feedback report")
        print(f"Total users who provided feedback: {report['total_users']}")
        for user_id, stats in report['user_stats'].items():
            print(f"User ID: {user_id}")
            print(f"  Mean Rating: {stats['mean']:.2f}")
            print(f"  Rating Standard Deviation: {stats['std_dev']:.2f}")

    def run_feedback_loop(self):
        """
        Runs the entire feedback loop: load, process, update model, and generate report.
        """
        feedback_data = self.load_feedback()
        if not feedback_data:
            logging.error("No feedback data available to process")
            return

        feedback_summary = self.process_feedback(feedback_data)
        user_data = self.load_user_data()

        if feedback_summary:
            self.update_model(feedback_summary, user_data)
            self.save_user_data(user_data)

            report = self.generate_feedback_report(feedback_summary)
            self.display_feedback_report(report)
        else:
            logging.info("No valid feedback to process")

if __name__ == "__main__":
    feedback_processor = FeedbackProcessor("feedback_store.json", "user_data_store.json")
    feedback_processor.run_feedback_loop()