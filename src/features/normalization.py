import torch


class GlobalFeatureNormalizer:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, feature_list):
        """
        feature_list:
            list of tensors with shape [11]
        """

        features = torch.stack(
            feature_list
        ).float()

        self.mean = features.mean(
            dim=0
        )

        self.std = features.std(
            dim=0,
            unbiased=False
        )

        # Prevent division by zero
        self.std = torch.where(
            self.std < 1e-8,
            torch.ones_like(self.std),
            self.std
        )

    def transform(self, features):
        """
        Normalize using training statistics.
        """

        if self.mean is None:
            raise RuntimeError(
                "Normalizer has not been fitted."
            )

        return (
            features - self.mean
        ) / self.std

    def fit_transform(self, feature_list):

        self.fit(
            feature_list
        )

        return [
            self.transform(
                feature
            )
            for feature in feature_list
        ]