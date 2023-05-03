import re

filepath='/Users/piercemaloney/Desktop/JP/IW/results/500_iterations/500_3player_goir_maybeshort.txt'

def extract_floats(filepath):
    floats_list = []

    with open(filepath, "r") as file:
        content = file.readlines()

    for line in content:
        match = re.search(r"Return: (\d+\.\d+)", line)
        if match:
            floats_list.append(float(match.group(1)))

    return floats_list[::2]

print("Floats extracted:", extract_floats(filepath))
