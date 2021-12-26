import os
# data_filename = os.path.join("data", 'penguin.csv')
with open("penguins.csv", "rt") as f:
    data_line = f.readlines()
    for line in data_line:
        print(line)
