import matplotlib.pyplot as plt

def read_data(filepath):
    """
    :param filename: the file to read data from - csv
    :return: the data in a list with ['text', 'label of text', 'abstractness tag']
    @author: Li Lin.
    """
    complete_data = []
    with open(filepath, encoding="utf8") as f:
        for lines in f:
            line = lines.strip("\n").split('\t')[:3]
            complete_data.append(line)

    return complete_data[1:]  # skip the header


class Abstractness_analysis(object):
    '''
    To analyze the abstractness tags of the pap dataset.
    The analysis mainly includes: unigram abstractness tag, bigram abstractness, tokens and abstractness
    @author: Li Lin.
    '''
    def __init__(self,filepath):
        self.filepath=filepath
        self.content=read_data(self.filepath)
        self.abstractness=[i[2] for i in self.content]
        self.abstractness_counts=self.count_abstractness()
        self.ab_mapping=self.ab_tags()
        self.ab_unigrams_counts = self.count_abstract_uni()
        self.ab_bigrams_counts = self.count_ab_bigrams()
        self.ab_tokens_mapping = self.get_ab_tokens_mapping()

    def count_abstractness(self):
        abstractness_counts={}
        for i in self.abstractness:
            if i in abstractness_counts:
                abstractness_counts[i]+=1
            else:
                abstractness_counts[i]=1
        abstractness_counts = dict(sorted(abstractness_counts.items(), key=lambda item: item[1], reverse=True))
        return abstractness_counts
    
    def ab_tags(self,num=2):
        if num==0:
            content=[i for i in self.content if i[1]=='implausible']
            print('get implausible data abstractness tag')
        elif num==1:
            content=[i for i in self.content if i[1]=='plausible']
            print('get plausible data abstractness tag')
        else:
            content=self.content
            print('get all data abstractness tag')

        ab_mapping=[]
        for i in content:
            token=i[0].split(' ')
            token_ab=i[2].split('-')
            ab_mapping.append([[t,j] for t,j in zip(token,token_ab)])
        return ab_mapping
    
    def count_abstract_uni(self,num=2):
        '''
        Count and sort the quantities for each abstractness tag separately.
        :return: a sorted set of abstractness and its count
        '''
        if num==0:
            content=self.ab_tags(0)
        elif num==1:
            content=self.ab_tags(1)
        else:
            content=self.ab_tags()


        result=[{},{},{}]

        for i in content:
            for j in range(len(i)):
                if i[j][1] not in result[j]:
                    result[j][i[j][1]]=1
                else:
                    result[j][i[j][1]]+=1
        result=[dict(sorted(i.items(), key=lambda item: item[1], reverse=True)) for i in result]
        return result
    
    def plot_ab_uni_distribution(self,num=2):
        '''
        Plot the distribution of pos tags for each class.
        '''
        if num==0:
            counts=self.count_abstract_uni(0)
        elif num==1:
            counts=self.count_abstract_uni(1)
        else:
            counts=self.count_abstract_uni() #self.ab_unigrams_counts
            
        labels1, values1 = zip(*counts[0].items())
        labels2, values2 = zip(*counts[1].items())
        labels3, values3 = zip(*counts[2].items())

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

    def count_ab_bigrams(self,num=2):
        """
            helper method to count ab bigrams per line
            :return: a dictionary of the form {bigram: count}
        """
        if num==0:
            content=self.ab_tags(0)
        elif num==1:
            content=self.ab_tags(1)
        else:
            content=self.ab_tags()

        ab_bigrams_dict={}
        temp_dict1={}
        temp_dict2={}
        temp_dict3={}
        temp_dict4={}
        for i in content:
            # join bigram as one string instead of list of two strings
            first_bigram = '-'.join([i[0][1], i[1][1]])
            second_bigram = '-'.join([i[1][1], i[2][1]])
            third_bigram = '-'.join([i[0][1], i[2][1]])
            temp_dict1[first_bigram] = temp_dict1.get(first_bigram, 0) + 1
            temp_dict2[second_bigram] = temp_dict2.get(second_bigram, 0) + 1
            temp_dict3[third_bigram] = temp_dict3.get(third_bigram, 0) + 1
            temp_dict4[first_bigram] = temp_dict4.get(first_bigram, 0) + 1
            temp_dict4[second_bigram] = temp_dict4.get(second_bigram, 0) + 1
            temp_dict4[third_bigram] = temp_dict4.get(third_bigram, 0) + 1
        ab_bigrams_dict['first_bigrams']=temp_dict1
        ab_bigrams_dict['second_bigrams']=temp_dict2
        ab_bigrams_dict['third_bigrams']=temp_dict3
        ab_bigrams_dict['total_bigrams']=temp_dict4
        return ab_bigrams_dict
    
    def plot_ab_bi_distribution(self,num=2):
        '''
        Plot the distribution of ab_bigram tags.
        '''
        if num==0:
            counts=self.count_ab_bigrams(0)
        elif num==1:
            counts=self.count_ab_bigrams(1)
        else:
            counts=self.count_ab_bigrams()

        labels1, values1 = zip(*counts['first_bigrams'].items())
        labels2, values2 = zip(*counts['second_bigrams'].items())
        labels3, values3 = zip(*counts['third_bigrams'].items())
        labels4, values4 = zip(*counts['total_bigrams'].items())
        max_value=max(max(values1),max(values2),max(values3),max(values4))

        fig, (ax1, ax2, ax3,ax4) = plt.subplots(1,4, figsize=(14, 1),gridspec_kw={'hspace': 0.35})
        ax1.bar(labels1, values1, color='blue')
        ax1.set_title('first_bigrams',fontsize=8)
        ax1.tick_params(axis='both', which='both', labelsize=8)
        ax1.set_yticks(range(0, max_value+ 1, max_value//5) ) 
        ax2.bar(labels2, values2, color='green')
        ax2.set_title('second_bigrams', fontsize=8)
        ax2.tick_params(axis='both', which='both', labelsize=8)
        ax2.set_yticks(range(0, max_value + 1, max_value//5)) 
        ax3.bar(labels3, values3, color='red')
        ax3.set_title('third_bigrams', fontsize=8)
        ax3.tick_params(axis='both', which='both', labelsize=8)
        ax3.set_yticks(range(0, max_value + 1, max_value//5)) 
        ax4.bar(labels4, values4, color='red')
        ax4.set_title('total_bigrams', fontsize=8)
        ax4.tick_params(axis='both', which='both', labelsize=8)
        ax4.set_yticks(range(0, max_value + 1, max_value//5))

        #plt.tight_layout()
        plt.show()

    def get_ab_tokens_mapping(self,num=2):
        '''
        Get the mapping of tokens and abstractness tags.
        :return: a list of list of tuples
        '''
        if num==0:
            content=self.ab_tags(0)
        elif num==1:
            content=self.ab_tags(1)
        else:
            content=self.ab_tags()

        tdict={}
        map={'a':0,'m':1,'c':2}
        for i in content:
            for j in i:
                if j[0] in tdict:
                    tdict[j[0]][map[j[1]]]+=1
                else:  
                    tdict[j[0]]=[0,0,0]
                    tdict[j[0]][map[j[1]]]+=1

        ab_token_dict={'a':{},'m':{},'c':{}} #abstractness token dictionary
        for i in tdict:
            if tdict[i][0]>0:
                ab_token_dict['a'][i]=tdict[i][0]
            elif tdict[i][1]>0:
                ab_token_dict['m'][i]=tdict[i][1]
            else:
                ab_token_dict['c'][i]=tdict[i][2]

        ab_token_dict['a']=dict(sorted(ab_token_dict['a'].items(), key=lambda item: item[1], reverse=True))
        ab_token_dict['m']=dict(sorted(ab_token_dict['m'].items(), key=lambda item: item[1], reverse=True))
        ab_token_dict['c']=dict(sorted(ab_token_dict['c'].items(), key=lambda item: item[1], reverse=True))
        return ab_token_dict
    
    def plot_ab_tokens_distribution(self,num=2):
        if num==0:
            counts=self.get_ab_tokens_mapping(0)
        elif num==1:
            counts=self.get_ab_tokens_mapping(1)
        else:
            counts=self.get_ab_tokens_mapping()
        
        fig, axs = plt.subplots(3,1, figsize=(15, 5), sharey=True)

        counts['a']=dict(list(counts['a'].items())[:10])
        counts['m']=dict(list(counts['m'].items())[:10])
        counts['c']=dict(list(counts['c'].items())[:10])
        for i, (key, values) in enumerate(counts.items()):
            ax = axs[i]
            sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)
            labels, heights = zip(*sorted_values)
            bars = ax.bar(labels, heights)
            
            for bar, label in zip(bars, labels):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')

            ax.set_title(f'Abstract tag: {key}')
            ax.set_xlabel('Words')
            ax.set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()
    
    def plot_ab_tokens_onlynumber_distribution(self,num=2):
        if num==0:
            counts=self.get_ab_tokens_mapping(0)
        elif num==1:
            counts=self.get_ab_tokens_mapping(1)
        else:
            counts=self.get_ab_tokens_mapping()
        counts['a']=dict(list(counts['a'].items())[:150])
        counts['m']=dict(list(counts['m'].items())[:150])
        counts['c']=dict(list(counts['c'].items())[:150])


        counts2={'a':[],'m':[],'c':[]}
        for i in counts2:
            for j in counts[i]:
                counts2[i].append(counts[i][j])

        x_values = range(1, len(counts2['a']) + 1)
        plt.figure(figsize=(7, 3))

        # Plotting trend lines for each dataset
        plt.plot(x_values, counts2['a'], marker='o', linestyle='-', label='a', markersize=2)
        plt.plot(x_values, counts2['m'], marker='o', linestyle='-', label='m', markersize=2)
        plt.plot(x_values, counts2['c'], marker='o', linestyle='-', label='c', markersize=2)

        # Adding legends and labels
        plt.legend()
        plt.title('Trend Chart')
        plt.xlabel('Word Index')
        plt.ylabel('Value')

        # Display the plot
        plt.show()
