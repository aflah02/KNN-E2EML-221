import os
import numpy as np
import random

sex_conversion = {"male": 0, "female": 1}
penguin_label_conversion = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}

def main():
    ls_penguins = load_data()
    ls_train, ls_test = split_data(ls_penguins)
    print(ls_train)
    print(ls_test)

def split_data(ls_penguins):
    ls_train = []
    ls_test = []
    ls_train = ls_penguins[0:len(ls_penguins)//2]
    ls_test = ls_penguins[len(ls_penguins)//2:]
    return ls_train, ls_test

def load_data():
    ls_penguins = []
    with open("penguins.csv", "rt") as f:
        data_line = f.readlines()
        for line in data_line[1:]:
            ls = []
            try:
                line_data = line.split(",")
                numerical_data = list(float(x.rstrip()) for x in line_data[2:6])
                ls.extend(numerical_data)
                sex_features = sex_conversion[line_data[6]]
                ls.append(sex_features)
                label = penguin_label_conversion[line_data[0]]
                ls.append(label)
                ls_penguins.append(ls)
            except ValueError:
                pass
            except KeyError:
                pass
            
    return ls_penguins

if __name__ == '__main__':
    main()