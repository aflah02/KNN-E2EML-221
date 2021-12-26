import os

def load_data():
    with open("penguins.csv", "rt") as f:
        data_line = f.readlines()
        for line in data_line[1:]:
            line_data = line.split(",")
            numerical_data = list(float(x.rstrip()) for x in line_data[2:6])
            sex_features = line_data[6]
            label = line_data[0]
            print(numerical_data, sex_features, label)

if __name__ == '__main__':
    load_data()