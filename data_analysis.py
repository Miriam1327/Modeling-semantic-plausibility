import nltk
from textstat import textstat
# nltk.download('averaged_perceptron_tagger')


def read_data(filename):
    """
    helper method to read the file line by line
    :param filename: the file to read data from - csv
    :return: the complete data in a list
    """
    complete_data = []
    # open the file with utf8 encoding, split it at the comma (it should be csv)
    with open(filename, encoding="utf8") as f:
        for lines in f:
            # the train file is separates by semicolons instead of commas
            if "train.csv" in filename:
                line = lines.strip("\n").split(';')
            else:
                line = lines.strip("\n").split(',')
            complete_data.append(line)

    return complete_data[1:]  # skip the header


class DataAnalysis:
    """
        This class provides methods to read-in and process the data from the input file

        The file was created on Mon Dec 4th 2023
            it was last edited on Tue Dec 5th 2023

        @author: Miriam S.
    """
    def __init__(self, filename):
        self.file_content = read_data(filename)
        self.all_tokens = self.extract_word_tokens()
        self.word_dict = self.store_words()
        self.num_unique_tokens = self.unique_word_count()
        self.total_token_count = self.total_word_count()
        self.readability_scores = self.readability()

    def dataset_statistics(self):
        """
        helper method to return number of rows and columns
        :return: a tuple of the form (num_rows, num_columns)
        """
        return len(self.file_content), len(self.file_content[0])

    def extract_word_tokens(self):
        """
            helper methods to extract the word tokens (non-unique)
            :return: a list containing all word tokens
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
        """
        return len(self.word_dict.keys())

    def plot_word_frequency(self, num_tokens):
        """
        helper method plot the frequency distribution of the first num_tokens tokens
        :param num_tokens: the number of tokens to plot the frequency of
        """
        f_dict = nltk.FreqDist(self.all_tokens)
        print("words appearing just once", len(f_dict.hapaxes()))
        f_dict.plot(num_tokens)

    def average_word_length(self):
        """
        this is a helper method to calculate the average word length
        :return: a float representing the average word length, rounded to two decimals
        """
        # join all tokens to form one string without spaces
        whole_string = ''.join(data.all_tokens)
        return '{0:.3g}'.format(len(whole_string)/self.total_word_count())

    def pos_tags(self):
        """
        helper method to assign POS tags to words in each phrase
        :return: a list containing lists of words + POS tags in tuples (per phase)
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
        helper method to calculate the average readability score (using flesch reading ease)
        :return: a float representing the average readability score
        """
        readability_sum = 0
        for val in self.readability_scores.values():
            readability_sum += val

        return '{0:.3g}'.format(readability_sum/len(self.readability().keys()))

    def plausibility_rating_count(self):
        """
        helper method to count the ratio of plausible and implausible ratings
        :return: a dictionary with plausible (1) and implausible (0) as keys, their corresponding counts as values
        """
        plausibility_dict = dict()
        for line in self.file_content:
            if line[0] in plausibility_dict.keys():
                plausibility_dict[line[0]] += 1
            else:
                plausibility_dict[line[0]] = 1
        return plausibility_dict

