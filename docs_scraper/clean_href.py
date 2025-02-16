import re

def extract_href_values(input_file, output_file):
    """
    Extracts the href values from <a> tags in an HTML file and writes them to a new file.

    Args:
        input_file: Path to the input HTML file.
        output_file: Path to the output file where href values will be written.
    """

    try:
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                # Use regular expression to find href values within <a> tags
                matches = re.findall(r'<a.*?href="([^"]*)".*?>', line)  #Improved regex to handle variations
                for match in matches:
                    outfile.write(match + '\n')

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage:
input_filename = '/home/vicrrs/projetos/meus_projetos/metropolis-rag-assistant/docs_scraper/deepstream_raw_html.txt'  # Replace with your input file name
output_filename = 'links.txt'  # Replace with your desired output file name

extract_href_values(input_filename, output_filename)

print(f"Href values extracted from '{input_filename}' and saved to '{output_filename}'")