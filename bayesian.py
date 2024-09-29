import pandas as pd
import numpy as np
from collections import Counter


class BayesianLabelPredictor:
    def __init__(self, dataset, smoothing=1):
        """
        Initialize the class with a dataset of daily labels and optional smoothing.
        :param dataset: A list or numpy array of daily labels (integers or categorical)
        :param smoothing: Smoothing factor for Laplace smoothing (default = 1)
        """
        self.GMM_labels = np.array(dataset['Group'])
        self.LLM_labels = np.array(dataset['LLM_Labels'])
        self.unique_labels_GMM = np.unique(self.GMM_labels)
        self.unique_labels_LLM = np.unique(self.LLM_labels)
        
        self.n = len(self.unique_labels_GMM)
        self.m = len(self.unique_labels_LLM)
        self.smoothing = smoothing
        self.today_GMM_label = self.GMM_labels[-1]
        self.today_LLM_label = self.LLM_labels[-1]

        # Initialize the prior matrix n x n (today's label on rows, tomorrow's on columns) with smoothing
        self.prior_matrix = np.zeros((self.m, self.n, self.n))

        # Calculate the prior matrix from the dataset
        self.calculate_prior()

    def calculate_prior(self):
        """
        Calculate the initial prior matrix (conditional probability matrix) from the dataset.
        """
        # Use Laplace smoothing
        self.prior_matrix = np.ones((self.m, self.n, self.n)) * self.smoothing

        # Go through each pair of consecutive labels (today, tomorrow) to compute probabilities
        for i in range(len(self.GMM_labels) - 1):
            today_label = self.GMM_labels[i]
            tomorrow_label = self.GMM_labels[i + 1]
            today_label_LLM = self.LLM_labels[i]

            # Map labels to indices in the matrix
            today_index = np.where(self.unique_labels_GMM == today_label)[0][0]
            tomorrow_index = np.where(
                self.unique_labels_GMM == tomorrow_label)[0][0]
            today_index_LLM = np.where(self.unique_labels_LLM == today_label_LLM)[0][0]
            self.prior_matrix[today_index_LLM][today_index][tomorrow_index] += 1

    def update_with_new_observation(self, tomorrow_GMM_label, tomorrow_LLM_label):
        """
        Update the prior matrix based on a new observation of today's and tomorrow's labels.
        :param tomorrow_label: Tomorrow's observed label
        """
        # Find the indices for today's and tomorrow's labels
        
        today_index = np.where(self.unique_labels_GMM == self.today_GMM_label)[0][0]
        today_index_LLM = np.where(self.unique_labels_LLM == self.today_LLM_label)[0][0]
        tomorrow_index = np.where(self.unique_labels_GMM == tomorrow_GMM_label)[0][0]
        tomorrow_index_LLM = np.where(self.unique_labels_LLM == tomorrow_LLM_label)[0][0]
        self.today_GMM_label = tomorrow_GMM_label
        self.today_LLM_label = tomorrow_LLM_label
        # Update the count for this transition in the prior matrix
        self.prior_matrix[today_index_LLM][today_index][tomorrow_index] += 1

        # probability of tomorrow's label based on todays
        tomorrow_count = self.prior_matrix[tomorrow_index_LLM][tomorrow_index]
        tomorrow_probability = tomorrow_count/sum(tomorrow_count)
        return tomorrow_probability

    def get_prior_matrix(self):
        """
        Get the current state of the prior matrix.
        :return: The current prior matrix (2D array)
        """
        return self.prior_matrix


if __name__ == '__main__':
    knowledge_before_date = '2016-01-01'
    smoothing_alpha = 2
    data = pd.read_csv('./clean_data/combined_full.csv', index_col=0)
    prior_knowledge = data.loc[data.index < knowledge_before_date]
    after = data.loc[data.index >= knowledge_before_date]
    # after = after.loc[after.index <= '2019-12-31']
    # print(after.shape)
    after_index = after.index
    # split dataset into prior, and update
    prior_knowledge = prior_knowledge[['Group', 'LLM_Labels']]
    after = after[['Group', 'LLM_Labels']].values

    # initialize Bayesian prior
    predictor = BayesianLabelPredictor(
        prior_knowledge, smoothing=smoothing_alpha)
    predictor.calculate_prior()
    # print(predictor.get_prior_matrix())

    # initialize a list of probability
    probability = []
    for i in range(len(after)):
        tomorrow_probs = predictor.update_with_new_observation(after[i][0], after[i][1])
        probability.append(tomorrow_probs)
    # print(tomorrow_probs)
    # print(predictor.get_prior_matrix())
    probabilities_df = pd.DataFrame(probability, index=after_index)
    # print(probabilities_df.shape)
    probabilities_df.to_json('./clean_data/priors.json')
