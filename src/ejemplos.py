def remove_low_variance_features(self, X: pd.DataFrame, threshold: float = 0.01):
        """
        Removes features with low variance (below a given threshold).

        Args:
            X (pd.DataFrame): Input features.
            threshold (float): The variance threshold for feature selection.

        Returns:
            pd.DataFrame: DataFrame with low variance features removed.
        """
        selector = VarianceThreshold(threshold=threshold)
        return pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])
