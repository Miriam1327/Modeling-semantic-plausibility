import matplotlib.pyplot as plt
import nltk

def read_data_of_both(filepath):
    """
    :param filepath: the file path
    :return: the complete data in a list with ['label of text', 'text']
    @author: Li Lin.
    """
    complete_data = []
    # open the file with utf8 encoding, split it at the comma (it should be csv)
    with open(filepath, encoding="utf8") as f:
        for lines in f:
            line = lines.strip("\n").split(',')
            if "pep-3k/" in filepath:
                complete_data.append(line)
            else:
                temp_line=[]
                if line[1] == "implausible":
                    temp_line.append('0')
                else:
                    temp_line.append('1')
                temp_line.append(line[0])
                complete_data.append(temp_line)
                
    return complete_data[1:]  # skip the header

def get_filenames_list(*args):
    '''
    param arges: the str of file names
    return: the complete data in a list
    '''
    filenames_list=[]
    for i in args:
        print('get file from:',i)
        filenames_list.append(i)
    return filenames_list

class DataAnalysisCompare(object):
    '''
    For analyzing pap and pep-3k datasets. mainly include:
        1.Basic information: number of data and binary classes.
        2.Tokens: number of total tokens, number of unique tokens, tokens pair.
        3.Pos analysis: number of unigram pos, number of bigram pos.
    @author: Li Lin.
    '''
    def __init__(self, *args):
        self.file_content = []
        filenames_list=get_filenames_list(*args)
        for i in filenames_list:
            self.file_content+=read_data_of_both(i)
        self.all_tokens = self.extract_word_tokens()
        self.word_dict = self.store_words()
        self.num_unique_tokens = len(self.word_dict.keys())
        # self.total_token_count = self.total_word_count()
        self.pos_mapping = self.pos_tags()
        self.unique_pos = {}
        self.pos_counts = self.count_pos_bigrams()
        self.classes_num = self.class_count()
        self.pos_bigrams_counts = self.pos_bi_count()
        self.pos_unigrams_counts = self.pos_uni_count()
        self.tokens_bigrams_dict = self.count_bi_tokens()
        self.dataset_statistics= [len(self.file_content), len(self.file_content[0])]

    def class_count(self):
        """
        Count the quantities for each category separately.
        :return: a set of class and its count
        """
        classes_num = {"0":0, "1":0}
        for i in self.file_content:
            classes_num[i[0]] += 1
        return classes_num
    
    def pos_bi_count(self):
        '''
        Count and sort the quantities for each pos tag separately.
        :return: a sorted set of bigram pos and its count
        '''
        result = dict(sorted(self.pos_counts.items(), key=lambda item: item[1], reverse=True))
        return result
    
    def pos_uni_count(self):
        '''
        Count and sort the quantities for each pos tag separately.
        :return: a sorted set of unigram pos and its count
        '''
        result=[{},{},{}]
        for i in self.pos_mapping:
            for j in range(len(i)):
                if i[j][1] not in result[j]:
                    result[j][i[j][1]]=1
                else:
                    result[j][i[j][1]]+=1
        result=[dict(sorted(i.items(), key=lambda item: item[1], reverse=True)) for i in result]
        return result
    
    def plot_pos_bi_distribution(self,n):
        labels, values = zip(*list(self.pos_bigrams_counts.items())[:10])

        # Plot the bar chart
        plt.bar(labels, values, color='blue')
        plt.xlabel('POS Tag Pairs')
        plt.ylabel('Count')
        plt.title('Top n POS Tag Pairs')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.tight_layout()

        # Show the plot
        plt.show()
    
    def plot_pos_uni_distribution(self):
        '''
        Plot the distribution of pos tags for each class.
        '''
        labels1, values1 = zip(*self.pos_unigrams_counts[0].items())
        labels2, values2 = zip(*self.pos_unigrams_counts[1].items())
        labels3, values3 = zip(*self.pos_unigrams_counts[2].items())

        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12, 1),gridspec_kw={'hspace': 0.35})
        ax1.bar(labels1, values1, color='blue')
        ax1.set_title('Subject',fontsize=8)
        ax1.tick_params(axis='both', which='both', labelsize=8)
        ax1.set_yticks(range(0, max(values1)+ 1, max(values1)//5) ) 

        ax2.bar(labels2, values2, color='green')
        ax2.set_title('Verb', fontsize=8)
        ax2.tick_params(axis='both', which='both', labelsize=8)
        ax2.set_yticks(range(0, max(values1) + 1, max(values1)//5))  

        ax3.bar(labels3, values3, color='red')
        ax3.set_title('Object', fontsize=8)
        ax3.tick_params(axis='both', which='both', labelsize=8)
        ax3.set_yticks(range(0, max(values1) + 1, max(values1)//5))  

        #plt.tight_layout()
        plt.show()

    def count_bi_tokens(self):
        content=[]
        for i in self.file_content:
            content.append(i[1].split(' '))

        tokens_bigrams_dict={}
        temp_dict1={}
        temp_dict2={}
        for i in content:
        # join bigram as one string instead of list of two strings
            first_bigram = '-'.join([i[0], i[1]])
            second_bigram = '-'.join([i[1], i[2]])
            temp_dict1[first_bigram] = temp_dict1.get(first_bigram, 0) + 1
            temp_dict2[second_bigram] = temp_dict2.get(second_bigram, 0) + 1
        tokens_bigrams_dict['s-v']=dict(sorted(temp_dict1.items(), key=lambda item: item[1], reverse=True))
        tokens_bigrams_dict['v-o']=dict(sorted(temp_dict2.items(), key=lambda item: item[1], reverse=True))
        return tokens_bigrams_dict
    
    def plot_tokens_bi_distribution(self,n):
        # Extract top n items from each category
        top_5_s_v = dict(sorted(self.tokens_bigrams_dict['s-v'].items(), key=lambda x: x[1], reverse=True)[:n])
        top_5_v_o = dict(sorted(self.tokens_bigrams_dict['v-o'].items(), key=lambda x: x[1], reverse=True)[:n])

        fig, axs = plt.subplots(2,1, figsize=(12, 6))

        # Subplot 1
        axs[0].bar(top_5_s_v.keys(), top_5_s_v.values(), color='skyblue')
        axs[0].set_title('S-V Pairs')
        axs[0].set_ylabel('Frequency')
        axs[0].set_xticklabels(top_5_s_v.keys(), rotation=45, ha='right')

        # Subplot 2
        axs[1].bar(top_5_v_o.keys(), top_5_v_o.values(), color='lightcoral')
        axs[1].set_title('V-O Pairs')
        axs[1].set_ylabel('Frequency')
        axs[1].set_xticklabels(top_5_v_o.keys(), rotation=45, ha='right')

        plt.tight_layout()
        plt.show()
        
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
    

