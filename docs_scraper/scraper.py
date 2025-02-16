import httpx
from bs4 import BeautifulSoup

url = "https://docs.nvidia.com/metropolis/deepstream/dev-guide/"

def save_filtered_li():
    try:
        with httpx.Client() as client:
            response = client.get(url)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        matching_links = soup.find_all('a', href=lambda href: href and href.startswith('text/'))

        li_elements = []
        for link in matching_links:
            li = link.find_parent('li')
            if li and li not in li_elements:
                li_elements.append(li)

        with open("filtered_li_elements.txt", "w", encoding="utf-8") as f:
            for li in li_elements:
                f.write(str(li) + "\n\n")

        print(f"Elementos <li> encontrados: {len(li_elements)}")
        print("Arquivo 'filtered_li_elements.txt' salvo com sucesso!")

    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    save_filtered_li()