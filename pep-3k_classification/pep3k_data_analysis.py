import matplotlib.pyplot as plt
import nltk
import numpy as np
from textstat import textstat

# nltk.download('averaged_perceptron_tagger')


def read_data(filename):
    """
        helper method to read the file line by line
        :param filename: the file to read data from - csv
        :return: the complete data in a list
        @author: Miriam S.
    """
    complete_data = []
    # open the file with utf8 encoding, split it at the comma (it should be csv)
    with open(filename, encoding="utf8") as f:
        for lines in f:
            # the train file is separates by semicolons instead of commas
            if ";" in lines:  # my train.csv is semicolon-separated, the file of my teammates is not
                line = lines.strip("\n").split(';')
            else:
                line = lines.strip("\n").split(',')
            complete_data.append(line)

    return complete_data[1:]  # skip the header


class DataAnalysis:
    """
        This class provides methods to read-in and process the data from the input file

        The file was created on Mon Dec 4th 2023
            it was last edited on Tue Dec 12th 2023

        @author: Miriam S.
    """
    def __init__(self, filename):
        self.file_content = read_data(filename)
        self.all_tokens = self.extract_word_tokens()
        self.word_dict = self.store_words()
        self.num_unique_tokens = self.unique_word_count()
        self.total_token_count = self.total_word_count()
        self.readability_scores = self.readability()
        self.pos_mapping = self.pos_tags()
        self.unique_pos = {}
        self.pos_counts = self.count_pos_bigrams()

    def dataset_statistics(self):
        """
            helper method to return number of rows and columns
            :return: a tuple of the form (num_rows, num_columns)
            @author: Miriam S.
        """
        return len(self.file_content), len(self.file_content[0])

    def extract_word_tokens(self):
        """
            helper methods to extract the word tokens (non-unique)
            :return: a list containing all word tokens
            @author: Miriam S.
        """
        token_list = []
        for line in self.file_content:
            # split the phrase content into several tokens
            token_list.append(line[1].split())
        # return a flattened version of the whole data to omit the nested lists
        flattened_list = [token for sublist in token_list for token in sublist]
        return flattened_list

    def store_words(self):
        """
            this is a helper method to store the words in a dictionary
            :return: a dictionary storing all words with their word counts
            @author: Miriam S.
        """
        word_dict = dict()
        for token in self.all_tokens:
            # if a word is already part of the dictionary, increase its count by 1
            # otherwise assign a count of 1
            if token in word_dict.keys():
                word_dict[token] += 1
            else:
                word_dict[token] = 1
        return word_dict

    def total_word_count(self):
        """
            this is a helper method to calculate the total word count
            :return: an integer representing the total word count
            @author: Miriam S.
        """
        total_word_count = 0
        for val in self.word_dict.keys():
            # add up the counts from each word in the dictionary
            total_word_count += self.word_dict[val]
        return total_word_count

    def unique_word_count(self):
        """
            this is a helper method to calculate the unique word count
            :return: an integer representing the unique word count
            @author: Miriam S.
        """
        return len(self.word_dict.keys())

    def plot_word_frequency(self, num_tokens):
        """
            helper method plot the frequency distribution of the first num_tokens tokens
            :param num_tokens: the number of tokens to plot the frequency of
            @author: Miriam S.
        """
        f_dict = nltk.FreqDist(self.all_tokens)
        # print("words appearing just once", len(f_dict.hapaxes()))  # for word occurrences
        f_dict.plot(num_tokens)

    def average_word_length(self):
        """
            this is a helper method to calculate the average word length
            :return: a float representing the average word length, rounded to two decimals
            @author: Miriam S.
        """
        # join all tokens to form one string without spaces
        whole_string = ''.join(self.all_tokens)
        return '{0:.3g}'.format(len(whole_string)/self.total_word_count())

    def pos_tags(self):
        """
            helper method to assign POS tags to words in each phrase
            :return: a list containing lists of words + POS tags in tuples (per phase)
            @author: Miriam S.
        """
        token_list, pos_list = [], []
        # join all tokens to form one string without spaces
        for line in self.file_content:
            # split the phrase content into several tokens
            token_list.append(line[1].split())

        for i in token_list:
            pos_list.append(nltk.pos_tag(i))
        return pos_list

    def readability(self):
        """
            helper method to calculate the readability scores per line (using flesch reading ease)
            :return: a dictionary storing the text followed by the corresponding readability score
            @author: Miriam S.
        """
        separate_lines = []
        readability_dict = dict()
        for line in self.file_content:
            # split the phrase content into several tokens
            separate_lines.append(line[1])

        # calculate the readability score
        for i in separate_lines:
            readability_dict[i] = textstat.flesch_reading_ease(i)

        return readability_dict

    def avg_readability(self):
        """
            helper method to calculate the average readability score (using Flesch reading ease)
            :return: a float representing the average readability score
            @author: Miriam S.
        """
        readability_sum = 0
        for val in self.readability_scores.values():
            readability_sum += val

        return '{0:.3g}'.format(readability_sum/len(self.readability().keys()))

    def plausibility_rating_count(self):
        """
            helper method to count the ratio of plausible and implausible ratings
            :return: a dictionary with plausible (1) and implausible (0) ratings as keys,
            their corresponding occurrence counts as values
            @author: Miriam S.
        """
        plausibility_dict = dict()
        for line in self.file_content:
            if line[0] in plausibility_dict.keys():
                plausibility_dict[line[0]] += 1
            else:
                plausibility_dict[line[0]] = 1
        return plausibility_dict

    def count_pos_bigrams(self):
        """
            helper method to count pos bigrams per line
            :return: a dictionary of the form {bigram: count}
            @author: Miriam S.
        """
        line_tags, pos, all_pos = [], [], []
        pos_dict = dict()
        # iterate over all (word, pos) tuples per line and store all possible values in a set + extract only pos values
        for val in self.pos_mapping:
            for tup in val:
                all_pos.append(tup[1])
                pos.append(tup[1])
            line_tags.append(pos)
            pos = []
        self.unique_pos = set(all_pos)  # store all unique pos tags in set

        # count tuple occurrences in file and store in dict
        for instance in line_tags:
            # join bigram as one string instead of list of two strings
            first_bigram = ' '.join(instance[:2])
            second_bigram = ' '.join(instance[1:])
            # check whether bigram is already in dict and increase or set count accordingly
            if first_bigram in pos_dict:
                pos_dict[first_bigram] += 1
            else:
                pos_dict[first_bigram] = 1

            if second_bigram in pos_dict:
                pos_dict[second_bigram] += 1
            else:
                pos_dict[second_bigram] = 1
        return pos_dict

    def plot_pos_dict(self, filename):
        """
            helper method to plot the occurrences of pos-pairs occurring consecutively
            :param filename: the filename to store the plot image at
            @author: Miriam S.
        """
        # remove key-value pairs which occur less than 10 times for greater visibility
        result = {key: self.pos_counts[key] for key in self.pos_counts.keys() if self.pos_counts[key] >= 10}

        # create a bar plot with labels and occurrence counts
        try:
            x = np.arange(len(result.keys()))
            plt.bar(x, list(result.values()))
            plt.xticks(x, list(result.keys()), rotation=90)
            plt.savefig(filename, bbox_inches='tight')
            print("File {} successfully printed".format(filename))
        except:
            print("An error occurred, the file could not be created")

