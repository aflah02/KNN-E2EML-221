import os

sex_conversion = {"male": 0, "female": 1}
penguin_label_conversion = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}
def load_data():
    with open("penguins.csv", "rt") as f:
        data_line = f.readlines()
        for line in data_line[1:]:
            try:
                line_data = line.split(",")
                numerical_data = list(float(x.rstrip()) for x in line_data[2:6])
                sex_features = sex_conversion[line_data[6]]
                label = penguin_label_conversion[line_data[0]]
            except ValueError:
                pass
            except KeyError:
                pass
            print(numerical_data, sex_features, label)

if __name__ == '__main__':
    load_data()