# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:11:17 2021

@author: fellnmr9
"""
def file_selector(data_dir,str_pattern):
    import os
    file_list=[]
    for sel_file in os.listdir(data_dir):
        print(sel_file)
        if sel_file.endswith(str_pattern):
            file_list.append(os.path.join(data_dir, sel_file))
    return file_list