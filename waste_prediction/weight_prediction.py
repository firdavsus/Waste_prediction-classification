import csv

def get_weight():
    with open('C:\Users\acer\Desktop\WALL-E\data\ready_data\lifts_with_people.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            return row
        
print(get_weight())