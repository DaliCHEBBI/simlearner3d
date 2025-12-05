import os
import glob
import re
import tifffile as tf




FILENAME="/var/data/MAChebbi/datasets/TRAINING_BLOCS_SELECTION/left.txt"

def write_list_into_file(list_of_names,filename):
    with open(filename, "a+") as f:
        for l in list_of_names:
            f.write(l+"\n")
        f.close()




FOLDERS=["/mnt/store-lidarhd/production/chantiers/_LidarExpress/_dali/TRAINING_SELECTION/Z1/GENERATED_DISPARITIES",
         "/mnt/store-lidarhd/production/chantiers/_LidarExpress/_dali/TRAINING_SELECTION/Z2/GENERATED_DISPARITIES",
         "/mnt/store-lidarhd/production/chantiers/_LidarExpress/_dali/TRAINING_SELECTION/Z6/GENERATED_DISPARITIES",
         "/mnt/store-lidarhd/production/chantiers/_LidarExpress/_dali/TRAINING_SELECTION/Z11/GENERATED_DISPARITIES",
         "/mnt/store-lidarhd/production/chantiers/_LidarExpress/_dali/TRAINING_SELECTION/Z13/GENERATED_DISPARITIES"]



def estimate_number_of_examples ( folders):
    number_of_examples = 0 
    for folder in folders:
        for root, dirs, _ in os.walk(folder):
            for dir in dirs:
                full_name_dir= os.path.join( root, dir)
                list_of_names_validated=[]
                list_left_image_names = [f for f in os.listdir(full_name_dir) if re.match(r'l_+.*\.tif', f)]
                # select names of left and right images that are of width = 1024
                for f in list_left_image_names:
                    full_name = os.path.join(full_name_dir,f)
                    imagel = tf.imread(full_name)
                    imager = tf.imread(full_name.replace("l_","r_"))
                    if (imagel.shape[-2:]==(1024,1024)) and (imager.shape[-2:]==(1024,1024)) :
                        list_of_names_validated.append(full_name)
                write_list_into_file(list_of_names_validated,FILENAME)
                number_of_examples+=len(list_of_names_validated)
        print(folder, number_of_examples)




if __name__=="__main__":
    estimate_number_of_examples(FOLDERS)