from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from pep3k_keras_new import ClassifierNew
from pep3k_keras import Classifier


class Evaluation:
    """
        This class provides methods to evaluate the performance of the classifier

        The file was created on     Thu Jan 11th 2024
            it was last edited on   Wed Jan 17th 2024

        @author: Miriam S.
    """
    def __init__(self, filename_train, filename_test, filename_dev,
                 dev_pap_bin, train_pap_bin):
        """
        this is the constructor for the class Evaluation containing several important variables
        it calls the class Classifier to access the classifier and the corresponding predictions
        :param filename_train: the name of the training file
        :param filename_test: the name of the test file
        :param filename_dev: the name of the development file
        :param dev_pap_bin: the name of the development file from pap (binary) used as additional data (dev.csv)
        :param train_pap_bin: the name of the training file from pap (binary) used as additional data (train.csv)
        @author: Miriam S.
        """
        self.classifier = ClassifierNew(filename_train, filename_test, filename_dev,
                                     dev_pap_bin, train_pap_bin)
        self.true_labels = self.classifier.labels_test
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


if __name__ == "__main__":
    eval = Evaluation("../pep-3k/train-dev-test-split/train.csv",
                      "../pep-3k/train-dev-test-split/test.csv",
                      "../pep-3k/train-dev-test-split/dev.csv",
                      "../binary/dev.csv",
                      "../binary/train.csv")
    print(eval.true_labels)
    print(eval.predicted_labels)
    print("prec", eval.precision())
    print("rec", eval.recall())
    print("f1", eval.f1_score())
    print("auc roc", eval.roc_auc())

