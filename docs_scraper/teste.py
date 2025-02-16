import httpx
import os
import asyncio
from urllib.parse import urlparse
from bs4 import BeautifulSoup

async def fetch_and_save_text(base_url, links_file, output_dir):

    try:
        with open(links_file, 'r', encoding='utf-8') as f:
            links = [line.strip() for line in f]

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        async with httpx.AsyncClient() as client:
            for link in links:
                full_url = base_url + link
                try:
                    response = await client.get(full_url)
                    response.raise_for_status()

                    parsed_url = urlparse(link)
                    filename = os.path.basename(parsed_url.path)
                    name, _ = os.path.splitext(filename)

                    soup = BeautifulSoup(response.text, 'html.parser')
                    text = soup.get_text(strip=True)

                    output_filename = os.path.join(output_dir, name + ".txt")
                    with open(output_filename, 'w', encoding='utf-8') as outfile:
                        outfile.write(text)

                    print(f"Text content from '{full_url}' saved to '{output_filename}'")

                except httpx.HTTPStatusError as e:
                    print(f"Error fetching URL '{full_url}': {e}")
                except httpx.RequestError as e:
                    print(f"Error fetching URL '{full_url}': {e}")
                except Exception as e:
                    print(f"An error occurred: {e}")

    except FileNotFoundError:
        print(f"Error: Links file '{links_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


async def main():
    base_url = "https://docs.nvidia.com/metropolis/deepstream/dev-guide/"
    links_filename = "/home/vicrrs/projetos/meus_projetos/metropolis-rag-assistant/docs_scraper/txts/links.txt"
    output_directory = "text_output"

    await fetch_and_save_text(base_url, links_filename, output_directory)

if __name__ == "__main__":
   asyncio.run(main())

print("Finished processing.")