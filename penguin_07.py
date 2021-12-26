import os
import numpy as np

sex_conversion = {"male": 0, "female": 1}
penguin_label_conversion = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}

def main():
    ls_penguins = load_data()
    for i in ls_penguins:
        print(i)

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
            except ValueError:
                pass
            except KeyError:
                pass
            ls_penguins.append(ls)
    return ls_penguins

if __name__ == '__main__':
    main()