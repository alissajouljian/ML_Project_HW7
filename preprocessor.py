# import numpy as np
# import pandas as pd

# class Preprocessor:
#     def __init__(self):
#         self.column_means = None
#         self.min_values = None
#         self.max_values = None

#         # Initialize target name
#         self.target_column = "In-hospital_death"

#     def fit(self, X):
#         # Calculate column means for filling NaN values
#         self.column_means = X.mean()

#         # Calculate min and max values for scaling
#         self.min_values = np.nanmin(X, axis=0)
#         self.max_values = np.nanmax(X, axis=0)

#     def transform(self, X):
#         # Fill NaN values with column means
#         # X_filled = np.where(np.isnan(X_none_to_nan), self.column_means, X_none_to_nan)
#         # X_filled = np.where(np.isnan(X), self.column_means, X)

#         # # Scale features using min-max scaling
#         # X_scaled = (X_filled - self.min_values) / (self.max_values - self.min_values)
#         # X_filled = np.where(np.isnan(X_scaled), self.column_means, X_scaled)


#         # Pandas initialization
#         X.fillna(self.column_means, inplace=True)

#         # TODO: Must be written scaling part:

#         ...

#         #################3

#         # Train mode checking and return
#         if self.target_column in X.columns:
#             X_transformed = X.drop(self.target_column, axis=1)
#             y = X[self.target_column]
#             return X_transformed, y

#         print(X.columns)
#         return X
#         ###################

#     def fit_transform(self, X):
#         self.fit(X)
#         return self.transform(X)

import numpy as np
import pandas as pd


class Preprocessor:
    def __init__(self):
        self.column_means = None
        self.min_values = None
        self.max_values = None

    def fit(self, df):
        self.column_means = df.mean()
        self.min_values = df.min()
        self.max_values = df.max()

    def transform(self, df, poly=False):
        # Fill NaN values with column means
        df_filled = df.fillna(self.column_means)  # , inplace=True

        # Scale features using min-max scaling
        df_scaled = (df_filled - self.min_values) / (self.max_values - self.min_values)

        if poly:
            # Identify non-boolean columns
            non_boolean_columns = df_scaled.nunique() > 2

            # Generate polynomial features for non-boolean columns
            poly_features = []
            for degree in [2, 3]:
                poly_features.append(df_scaled.loc[:, non_boolean_columns].pow(degree))
            poly_features_df = pd.concat(poly_features, axis=1)

            # Concatenate the generated polynomial features with the original scaled data
            df_extended = pd.concat([df_scaled, poly_features_df], axis=1)
        else:
            df_extended = df_scaled

        # Go to pipeline code, will be deleted
        # Train mode checking
        # if self.target_column in df.columns:
        #     df_extended = df.drop(self.target_column, axis=1)
        #     y = df[self.target_column]
        #     return df_extended, y

        return df_extended