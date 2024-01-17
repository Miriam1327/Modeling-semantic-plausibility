import numpy as np
from collections import Counter
from keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from pep3k_data_analysis import DataAnalysis


class Classifier:
    """
        This class provides methods to read-in and process the data from the input file
        The file was created on     Wed Jan 10th 2024
            it was last edited on   Wed Jan 17th 2024
        @author: Miriam S.
    """
    def __init__(self, filename_train, filename_test, filename_dev,
                 pap_dev_bin, pap_train_bin):
        """
        this is the constructor for the class Classifier containing several important variables
        it calls the class DataAnalysis to access the general file contents
        :param filename_train: the name of the training file from pep3k (train.csv)
        :param filename_test: the name of the test file from pep-3k (test.csv)
        :param filename_dev: the name of the development file from pep-3k (dev.csv)
        :param pap_dev_bin: the name of the development file from pap (binary) used as additional data (dev.csv)
        :param pap_train_bin: the name of the training file from pap (binary) used as additional data (train.csv)
        @author: Miriam S.
        """
        # get the data from the DataAnalysis class and store it in variables
        # pep-3k
        self.data_train = DataAnalysis(filename_train)
        self.data_test = DataAnalysis(filename_test)
        self.data_dev = DataAnalysis(filename_dev)
        # binary pap
        self.data_dev_pap_bin = DataAnalysis(pap_dev_bin)
        self.data_train_pap_bin = DataAnalysis(pap_train_bin)
        # extract labels from pep-3k and store them in variables
        self.labels_test = self.extract_data(filename_test)[0]
        self.labels_train = self.extract_data(filename_train)[0]
        self.labels_dev = self.extract_data(filename_dev)[0]
        # extract texts from pep-3k and pap and store them in variables
        self.text_train = self.extract_data(filename_train)[1] + self.prepare_pap_bin(pap_train_bin)[1]
        self.text_dev = self.extract_data(filename_dev)[1] + self.prepare_pap_bin(pap_dev_bin)[1]
        self.text_test = self.extract_data(filename_test)[1]
        # extract labels from pap (binary and multiclass) and store them in variables
        self.labels_train_pap_bin = self.prepare_pap_bin(pap_train_bin)[0]
        self.labels_dev_pap_bin = self.prepare_pap_bin(pap_dev_bin)[0]

    def extract_data(self, filename):
        """
        method to extract texts and labels from the input data in the different files
        :param filename: the name of the corresponding file (either train.csv, test.csv, or dev.csv)
        :return: a tuple containing two separate lists with labels and texts
        @author: Miriam S.
        """
        all_texts, all_labels = [], []
        # check for the correct file name, extract the second part from the list (text) and append it to a new list
        # extract the first part of the list (label) and append it to a list as an integer
        if "train.csv" in filename:
            for i in self.data_train.file_content:
                # labels are converted into integers to facilitate usage (e.g., f1 calculation)
                all_labels.append(int(i[0]))
                all_texts.append(i[1])
        if "test.csv" in filename:
            for i in self.data_test.file_content:
                all_labels.append(int(i[0]))
                all_texts.append(i[1])
        if "dev.csv" in filename:
            for i in self.data_dev.file_content:
                all_labels.append(int(i[0]))
                all_texts.append(i[1])
        return all_labels, all_texts

    def prepare_pap_bin(self, filename):
        """
        method to extract texts and labels from the pap binary dataset in the different files
        this data is used for additional training data
        :param filename: the name of the corresponding file from pap (either train.csv, test.csv, or dev.csv)
        :return: a tuple containing two separate lists with labels and texts
        @author: Miriam S.
        """
        all_texts_pap, all_labels_pap = [], []
        # check for the correct file name,
        # extract the second part from the list (original_label) and append it to a new list
        # extract the first part from the list (text) and append it to a new list
        if "train.csv" in filename:
            for i in self.data_train_pap_bin.file_content:
                all_labels_pap.append(i[1])
                all_texts_pap.append(i[0])
        if "dev.csv" in filename:
            for i in self.data_dev_pap_bin.file_content:
                all_labels_pap.append(i[1])
                all_texts_pap.append(i[0])
        # pap labels are assigned to "implausible" and "plausible"
        # these are converted to 0 and 1 respectively to facilitate computation
        all_labels_pap = np.array(list(map(lambda label: 0 if label == 'implausible' else 1, all_labels_pap)))
        return all_labels_pap, all_texts_pap

    def word_counter(self):
        """
        method to count occurrences of individual words in the training instances
        needed for the tokenizer step, where embeddings are assigned based on number of words
        :return: a counter object (key-value pairs: word - num of occurrences)
        @author: Miriam S.
        """
        count = Counter()
        # iterate over texts and individual words, increase counter for each word
        for text in self.text_train:
            for word in text.split():
                count[word] += 1
        return count

    def predict_plausibility(self):
        """
        function to predict plausibility label based on a sequential keras model
        :return: a list containing the predicted labels (plausible or implausible)
        @author: Miriam S.
        """
        # data is prepared to use data from pap as additional training data
        # train and dev labels are concatenated as numpy arrays
        # train pep-3k and pap as well as dev data are appended for additional data
        train_labels = np.concatenate((np.array(self.labels_train),
                                       np.array(self.labels_train_pap_bin)))
        dev_labels = np.concatenate((np.array(self.labels_dev),
                                     np.array(self.labels_dev_pap_bin)))
        train_seqs = self.text_train
        dev_seqs = self.text_dev

        # get word frequencies on the train sentences
        counter = self.word_counter()
        num_words = len(counter)
        # the tokenizer is fit on the text from the training sentences which is used for the embedding layers
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(train_seqs)
        # all data instances are converted into embeddings of the same size
        # prepare train data - encode into embedding
        train_seqs = tokenizer.texts_to_sequences(train_seqs)
        train_padded = pad_sequences(train_seqs)
        # prepare validation data
        dev_seqs = tokenizer.texts_to_sequences(dev_seqs)
        dev_padded = pad_sequences(dev_seqs)

        # prepare test data (data for which we want to make predictions later)
        test_seqs = tokenizer.texts_to_sequences(self.text_test)
        test_padded = pad_sequences(test_seqs)

        # create a sequential model with the following components
        # an embedding layer with 32 dimensions
        # a bidirectional LSTM layer with 256 dimensions and a dropout of 0.5
        # a bidirectional LSTM layer with 64 dimensions and a dropout of 0.5
        # a unidimensional dense layer with sigmoid activation function
        # the dense layer includes an L1L2 kernel regularizer (applies a penalty on the layer's kernel)
        # an L2 bias regularizer (applies a penalty on the layer's bias)
        # an L2 activity regularizer (applies a penalty on the layer's output)
        model = Sequential([
            Embedding(num_words, 32),
            Bidirectional(LSTM(256, dropout=0.5, return_sequences=True)),
            Bidirectional(LSTM(64, dropout=0.5)),
            Dense(1, activation='sigmoid',
                  kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.L2(1e-4),
                  activity_regularizer=regularizers.L2(1e-5))
        ])
        # the model is compiled with binary cross entropy loss and the adam optimizer
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # the model is fit with the training and validation data using both texts and labels
        # model is fit using 35 epochs
        model.fit(
            train_padded, train_labels, epochs=35, validation_data=(dev_padded, dev_labels)
        )

        # get the predictions on the test set and return them as a sequence of 0/1
        # labels are chosen based on assigned probability: >= 0.5 means plausible (1) and < 0.5 means implausible (0)
        predictions = model.predict(test_padded)
        predictions = [1 if pr >= 0.5 else 0 for pr in predictions]

        return predictions