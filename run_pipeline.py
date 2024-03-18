# TODO: requirements.txt
import argparse
import joblib
import json
import pandas as pd

from model import Model
from preprocessor import Preprocessor


class Pipeline:
    """
    Pipeline
    =====================
    The Pipeline class provides a way to preprocess and model data for training and testing. 

    Parameters:
    ---------------------
    algorithm (str): The name of the algorithm to use for modeling.

    Attributes:
    ---------------------
    model_filename (str): The filename for saving the model in test mode.
    preprocessor_filename (str): The filename for saving the preprocessor in test mode.
    predictions_filename (str): The filename for saving the predictions in test mode.
    model (Model): The model object.
    preprocessor (Preprocessor): The preprocessor object.

    Methods:
    ---------------------
    run(data: pd.DataFrame, test: bool = False):
        Preprocesses and models the input data.

        Parameters:
        ---------------------
        data (pandas DataFrame): The input data to preprocess and model.
        test (bool): If True, load the preprocessor and model from their saved files and 
            use them to predict the output for the input data. Otherwise, fit the preprocessor 
            and model to the input data and save them to their respective files for future testing.

        Returns:
        ---------------------
        If test is True, a JSON file containing the predicted probabilities and threshold. Otherwise, nothing is returned.
    """

    def __init__(self, algorithm):
        """
        Initializes the Pipeline class.

        Parameters:
        ---------------------
        algorithm (str): The name of the algorithm to use for modeling.
        """
        self.model_filename = "model.sav" # Test mode
        self.preprocessor_filename = "preprocessor.sav" # Test mode
        self.predictions_filename = "predictions.json"
        self.target_column = "In-hospital_death"

        self.model = Model(algorithm)
        self.preprocessor = Preprocessor()

    def run(self, data, test=False):
        """
        Preprocesses and models the input data.

        Parameters:
        ---------------------
        data (pandas DataFrame): The input data to preprocess and model.
        test (bool): If True, load the preprocessor and model from their saved files and use them 
            to predict the output for the input data. Otherwise, fit the preprocessor and model to 
            the input data and save them to their respective files for future testing.

        Returns:
        ---------------------
        If test is True, a JSON file containing the predicted probabilities and threshold. Otherwise, nothing is returned.
        """
        if test:
            # Model and Preprocessor loading process.
            self.model = joblib.load(self.model_filename)
            self.preprocessor = joblib.load(self.preprocessor_filename)

            # Preprocessing and get predictions
            X = self.preprocessor.transform(data)
            
            # Get probabilties and best threshold
            predictions = self.model.predict_proba(X)
            
            predictions = predictions.tolist()

            threshold = self.model.threshold

            # Make dictionaries for saving.
            json_file = {
                "predict_probas": predictions, 
                "threshold": threshold,
            }

            # Saving JSON file.
            with open(self.predictions_filename, "w") as f:
                json.dump(json_file, f)

        else:
            # Separation of data and target
            X = data.drop(self.target_column, axis=1)
            y = data[self.target_column]

            # Preprocessor and model fitting.
            self.preprocessor.fit(X)
            X = self.preprocessor.transform(X)

            # Model fitting
            self.model.fit(X, y)

            # Saving fitted model and preprocessor.
            joblib.dump(self.model, self.model_filename)
            joblib.dump(self.preprocessor, self.preprocessor_filename)


def main():
    """
    main()
    --------------------
    The main() function serves as the entry point of the program.

    It uses the argparse module to parse command line arguments and the preprocessor 
    module from preprocessor.py to preprocess the data. Additionally, 
    it uses the Model module from model.py to train the model.

    If the program is run in test mode, it imports the pre-trained 
    model and generates predictions using the imported model.
    """

    # Define argparse.ArgumentParse object for argument manipulating
    parser = argparse.ArgumentParser(
        description="""
        It defines two arguments that can be passed to the program:

        1. "--data_path": a required argument that 
            specifies the path to the data file.

        2. "--inference": an optional argument of type bool 
            that activates test mode if set to "True".
        """,    
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Add --data_path and --inference arguments to parser
    parser.add_argument("--data_path", help="Path to data file.", required=True)
    parser.add_argument("--inference", help="Test mode activation.", required=False, default=False)
    parser.add_argument("--algorithm", help="Training model", required=False, default="GBoost")

    # --inference -> default = False. Default activates train mode.

    # Get arguments as dictionary from parser
    args = parser.parse_args() # returns dictionary-like object

    possible_falses = ["0", "false", "False"]

    path_of_data = args.data_path
    test_mode = args.inference not in possible_falses
    algorithm = args.algorithm

    # Reading CSV file
    DataFrame = pd.read_csv(path_of_data)

    # Pipeline running
    pipeline = Pipeline(algorithm)
    pipeline.run(DataFrame , test=test_mode)


if __name__ == "__main__":
    main()