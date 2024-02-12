from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from pep3k_keras_data_prep import Classifier


class EvaluationAdapted:
    """
        This class provides methods to evaluate the performance of the classifier for adapted data

        The file was created on     Wed Jan 24th 2024
            it was last edited on   Mon Jan 29th 2024

        @author: Miriam S.
    """
    def __init__(self, filename_train, filename_test, filename_dev,
                 dev_pap_bin, train_pap_bin, test_pap_bin, file='pep'):
        """
        this is the constructor for the class EvaluationAdapted containing several important variables
        it calls the class Classifier to access the classifier and the corresponding predictions
        :param filename_train: the name of the training file
        :param filename_test: the name of the test file
        :param filename_dev: the name of the development file
        :param dev_pap_bin: the name of the development file from pap (binary) used as additional data (dev.csv)
        :param train_pap_bin: the name of the training file from pap (binary) used as additional data (train.csv)
        :param test_pap_bin: the name of the test file from pap (binary) (test.csv)
        :param file: name of the file used as test
                     can be 'pep' (default) where s-v-o triples have NO third-person singular s for the verb
                     can be 'pap' where s-v-o triples have third-person singular s for the verb
        @author: Miriam S.
        """
        # make sure file is passed to other variables/methods
        self.file = file
        self.classifier = Classifier(filename_train, filename_test, filename_dev,
                                     dev_pap_bin, train_pap_bin, test_pap_bin, file=self.file)
        if self.file == 'pep':
            # use labels from pep
            self.true_labels = self.classifier.labels_test
        else:
            # use labels from pap
            self.true_labels = self.classifier.labels_test_pap_bin
        self.predicted_labels = self.classifier.predict_plausibility()

    def precision(self):
        """
        this is a method to calculate the precision score given predicted and true labels
        :return: the calculated precision score
        @author: Miriam S.
        """
        return precision_score(self.true_labels, self.predicted_labels)

    def recall(self):
        """
        this is a method to calculate the recall score given predicted and true labels
        :return: the calculated recall score
        @author: Miriam S.
        """
        return recall_score(self.true_labels, self.predicted_labels)

    def f1_score(self):
        """
        this is a method to calculate the f1 score given predicted and true labels
        :return: the calculated f1 score
        @author: Miriam S.
        """
        return f1_score(self.true_labels, self.predicted_labels)

    def roc_auc(self):
        """
        this is a method to calculate the roc auc score given predicted and true labels
        :return: the calculated roc auc score
        @author: Miriam S.
        """
        return roc_auc_score(self.true_labels, self.predicted_labels)

