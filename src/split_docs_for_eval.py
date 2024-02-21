import csv
import argparse
parser = argparse.ArgumentParser(description='Split documents into sentences for evaluation')
parser.add_argument('--input_file', type=str, help='Input file')
parser.add_argument('--output_file', type=str, help='Output file')

def split_docs_for_eval(input_file, output_file):
    # split each row by punctuation and write to output file
    with open(input_file, 'r') as f:
        with open(output_file, 'w') as o:
            reader = csv.reader(f, delimiter='\t')
            writer = csv.writer(o, delimiter='\t')
            for row in reader:
                for sentence in row[0].split('.'):
                    writer.writerow([sentence])

    

args = parser.parse_args()
split_docs_for_eval(args.input_file, args.output_file)

