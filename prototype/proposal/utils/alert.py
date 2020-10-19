import os

def should_overwrite_model(path):
    if os.path.isdir(path):
        while(1):
            ans = input("That model already exists. Do you want to overwrite? (y/n): ")
            if ans == "y" or ans == "yes":
                return True
            elif ans == "n" or ans == "no":
                return False
    else:
        return True
