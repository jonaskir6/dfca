import pickle
import csv

# Path to the pickle file
pickle_file_path = 'path/to/your/pickle_file.pkl'

# Open the pickle file in binary read mode
with open("results.pickle", 'rb') as file:
    # Load the data from the pickle file
    data = pickle.load(file)

with open('results.txt', 'w') as txt_file:
    txt_file.write(str(data))

with open('results.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    # If data is a list of dictionaries
    if isinstance(data, list) and isinstance(data[0], dict):
        # Write the header
        writer.writerow(data[0].keys())
        # Write the data rows
        for row in data:
            writer.writerow(row.values())
    
    # If data is a list of lists
    elif isinstance(data, list) and isinstance(data[0], list):
        # Write the data rows
        writer.writerows(data)
    
    # Handle other data structures as needed
    else:
        # Convert data to a string and write it as a single cell
        writer.writerow([str(data)])

# Print the loaded data
# print(data)