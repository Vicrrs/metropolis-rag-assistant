import re

def extract_href_values(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                matches = re.findall(r'<a.*?href="([^"]*)".*?>', line)
                for match in matches:
                    outfile.write(match + '\n')

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


input_filename = '/home/vicrrs/projetos/meus_projetos/metropolis-rag-assistant/docs_scraper/deepstream_raw_html.txt'  # Replace with your input file name
output_filename = 'links.txt'

extract_href_values(input_filename, output_filename)

print(f"Href values extracted from '{input_filename}' and saved to '{output_filename}'")