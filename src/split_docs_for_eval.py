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
                # split by ., !, or ?
                sentences = row[0].split('.')
                sentences = [s.strip() for s in sentences if s.strip()]
                for s in sentences:
                    writer.writerow([s])
                    

    

args = parser.parse_args()
split_docs_for_eval(args.input_file, args.output_file)

