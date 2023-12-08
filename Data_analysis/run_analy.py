
from data_analysis import *
import os


def get_filepath(name,classnum_name,file_name):
    
    current_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir,os.pardir))
    return os.path.abspath(os.path.join(parent_dir,name,'train-dev-test-split',classnum_name,file_name+'.csv'))



pepdev_path=get_filepath('pep-3k','','dev')
print(pepdev_path)
pepdev_data=read_data(pepdev_path)
print(pepdev_data)
print(type(pepdev_data))
