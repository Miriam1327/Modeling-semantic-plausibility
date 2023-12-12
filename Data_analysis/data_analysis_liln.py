from data_analysis import DataAnalysis
import os
import matplotlib.pyplot as plt
import nltk
import numpy as np

def read_data2(filename):
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
            line = lines.strip("\n").split(',')
            if "pap" in filename:
                temp_line=[]
                if line[1] == "implausible":
                    temp_line.append('0')
                else:
                    temp_line.append('1')
                temp_line.append(line[0])
                complete_data.append(temp_line)
            else:
                complete_data.append(line)

    return complete_data[1:]  # skip the header

class DataAnalysis2(DataAnalysis):
    def __init__(self, filenames_list):
        self.file_content = []
        for i in filenames_list:
            self.file_content+=read_data2(i)
        self.all_tokens = self.extract_word_tokens()
        self.word_dict = self.store_words()
        self.num_unique_tokens = self.unique_word_count()
        self.total_token_count = self.total_word_count()
        self.readability_scores = self.readability()
        self.pos_mapping = self.pos_tags()
        self.unique_pos = {}
        self.pos_counts = self.count_pos_bigrams()
        self.classes_num = self.class_count()
        self.pos_bigrams_counts = self.pos_bi_count()
        self.pos_unigrams_counts = self.pos_uni_count()
    
    def class_count(self):
        """
        Count the quantities for each category separately.
        :return: a set of class and its count
        """
        classes_num = {"0":0, "1":0}
        for i in self.file_content:
            classes_num [i[0]] += 1
        return classes_num
    
    def pos_bi_count(self):
        '''
        Count and sort the quantities for each pos tag separately.
        :return: a sorted set of pos and its count
        '''
        result = dict(sorted(self.pos_counts.items(), key=lambda item: item[1], reverse=True))
        return result
    
    def pos_uni_count(self):
        '''
        Count and sort the quantities for each pos tag separately.
        :return: a sorted set of pos and its count
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
        plt.title('Top 10 POS Tag Pairs')
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
        ax1.set_yticks(range(0, max(values1)+ 1, max(values1)//5) ) # Set y-axis ticks to intervals of 200

        ax2.bar(labels2, values2, color='green')
        ax2.set_title('Verb', fontsize=8)
        ax2.tick_params(axis='both', which='both', labelsize=8)
        ax2.set_yticks(range(0, max(values1) + 1, max(values1)//5))  # Set y-axis ticks to intervals of 200

        ax3.bar(labels3, values3, color='red')
        ax3.set_title('Object', fontsize=8)
        ax3.tick_params(axis='both', which='both', labelsize=8)
        ax3.set_yticks(range(0, max(values1) + 1, max(values1)//5))  # Set y-axis ticks to intervals of 200

        #plt.tight_layout()
        plt.show()


def get_filepath(name,classnum_name,file_name):
    '''
    param name: name of the dataset, pep-3k or pap
    param classnum_name: for pap dataset, binary or multiclass
    param file_name: dev, test, train
    return: the str of apath of the file
    '''
    current_dir =os.path.realpath(".")
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    filepath=os.path.abspath(os.path.join(parent_dir,'Data',name,'train-dev-test-split',classnum_name,file_name+'.csv'))
    return filepath

def get_dataset(*args):
    '''
    param arges: the str of file names
    return: the complete data in a list
    '''
    filenames_list=[]
    for i in args:
        print('get file from:',i)
        filenames_list.append(i)
    data_set=DataAnalysis2(filenames_list)
    return data_set