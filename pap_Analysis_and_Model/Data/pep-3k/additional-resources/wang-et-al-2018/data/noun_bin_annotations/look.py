import pickle

# Open the pickled file for reading in binary mode
with open('noun_bin_annotations/noun2sentience.p', 'rb') as file: #noun2masscount,noun2phase,noun2rigidity,noun2sentience,
    # Load the pickled data
    data = pickle.load(file)

# Now, 'data' contains the deserialized content of the pickled file
print(type(data))

list=[]
for i,j in data.items():
    #print(i,j)
    if j not in list:
        list.append(j)
    
print(list)
