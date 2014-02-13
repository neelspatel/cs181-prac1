import csv
import math

def round(value):
	print value
	decimal = value - math.floor(value)

	if decimal <= 0.4:
		return math.floor(value)
	if decimal >= 0.6: 
		return math.floor(value) + 1
	else:
		return math.floor(value) + 0.5


input_filename = "pred-cf-user-mean.csv"
output_filename = "rounded-pred-cf-user-mean.csv"

with open(input_filename, 'rb') as input_file:
	 with open(output_filename, 'wb') as output_file:
		writer = csv.writer(output_file, quoting=csv.QUOTE_MINIMAL)
		reader = csv.reader(input_file)

		header = next(reader, None)   
		writer.writerow(header)

		for row in reader:
			id = int(row[0])
			pred = round(float(row[1]))        	

			writer.writerow([id, pred]) 
       