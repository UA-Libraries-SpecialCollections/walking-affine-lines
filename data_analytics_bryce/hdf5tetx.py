
import os
import hdf5reader
from tkinter import filedialog

file = r"S:\Digital Projects\Encoding\testing\complete_hdf5_files\u0002_0000007.range_48_shelf_1_hdf5.h5"
os.chdir(r"C:\Users\dslocal\Desktop\Bryce Testing")


# with open ('analytics_output.txt', 'w', encoding='utf-8') as f:
    
#     y = hdf5reader.hdf5_read_analytics(file)
#     for key in y:      
#         f.write(key + ' : ' + str(y[key]) + '\n')
#     f.close()


# values = []
y = hdf5reader.hdf5_read_analytics(file)
# print(y)
# for value in y.values(): 
#     values.append(value)
# keys = []
x = hdf5reader.hdf5_segments(file)
# for key in x: 
#     keys.append(key) 

overall_dict= {dict(zip(x.keys(), y.values()))} 
print(overall_dict)




# with open ('segment_output.txt', 'r', encoding='utf-8') as f:

#     # x = hdf5reader.hdf5_segments(file)
#     # for key in x:      
#     #     f.write(key + ' : ' + str(x[key]) + '\n')
#     # f.close()   

'pip install "s:/Digital Projects/Encoding/testing/hdf5reader"'