from rdflib import Graph
from pathlib import Path
import re
import requests

def extract_csv_links_from_rdf(rdf_path: str):
    """Wczytuje RDF i zwraca zestaw linków .csv"""
    try:
        rdf_file = Path(rdf_path)
        print(f"📖 Odczytywanie RDF: {rdf_file.resolve()}")

        if not rdf_file.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku RDF: {rdf_file}")

        g = Graph()
        g.parse(str(rdf_file))

        csv_links = set()
        for _, _, o in g:
            obj_str = str(o)
            if obj_str.lower().endswith(".csv"):
                csv_links.add(obj_str)

        print(f"✅ Znaleziono {len(csv_links)} linków CSV w RDF")
        return csv_links

    except Exception as e:
        print(f"❌ Błąd podczas czytania RDF: {e}")
        return set()


def extract_csv_links_from_html(html_path: str):
    """Wczytuje HTML i zwraca zestaw linków .csv"""
    try:
        html_file = Path(html_path)
        print(f"📖 Odczytywanie HTML: {html_file.resolve()}")

        if not html_file.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku HTML: {html_file}")

        csv_pattern = re.compile(r'href="([^"]*\.csv)"', re.IGNORECASE)
        csv_links = set()

        with open(html_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                for match in csv_pattern.finditer(line):
                    href = match.group(1)
                    if href.startswith("http"):
                        csv_links.add(href)
                    else:
                        # możesz zmienić tę bazę URL, jeśli Twoje linki są inne
                        full_url = f"https://road.affectivese.org/datasets/GraphNeuralNetwork/{href}"
                        csv_links.add(full_url)

        print(f"✅ Znaleziono {len(csv_links)} linków CSV w HTML")
        return csv_links

    except Exception as e:
        print(f"❌ Błąd podczas czytania HTML: {e}")
        return set()


def download_csv_files(links: set[str], output_folder: str = "csv_downloads"):
    """Pobiera wszystkie pliki CSV z podanych linków"""
    if not links:
        print("⚠️ Brak linków do pobrania.")
        return

    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(links)
    print(f"⬇️ Rozpoczynam pobieranie {len(links)} plików CSV do folderu: {output_dir.resolve()}")

    for i, link in enumerate(sorted(links), start=1):
        try:
            filename = Path(link).name
            save_path = output_dir / filename

            print(f"\n📥 [{i}/{total}] Pobieranie: {link}")
            response = requests.get(link, timeout=30)
            response.raise_for_status()

            with open(save_path, "wb") as f:
                f.write(response.content)

            print(f"✅ [{i}/{total}] Zapisano: {save_path}")

        except Exception as e:
            print(f"❌ [{i}/{total}] Błąd pobierania {link}: {e}")

    print("\n🎉 Zakończono pobieranie wszystkich plików.")


def compare_and_download(rdf_path: str, html_path: str):
    """Porównuje linki CSV z RDF i HTML, a następnie pobiera wszystkie"""
    print("\n🔍 Porównywanie linków CSV...\n")

    rdf_links = extract_csv_links_from_rdf(rdf_path)
    html_links = extract_csv_links_from_html(html_path)

    all_links = rdf_links | html_links  # połączenie obu zbiorów (unikalne)
    print(f"\n📊 Podsumowanie:")
    print(f"- Linki w RDF:  {len(rdf_links)}")
    print(f"- Linki w HTML: {len(html_links)}")
    print(f"- Razem unikalnych linków: {len(all_links)}")

    # Pobierz wszystkie pliki CSV
    download_csv_files(all_links)


if __name__ == "__main__":
    # 👇 Wystarczy, że zmienisz te dwie ścieżki (względne lub bezwzględne)
    rdf_file = "inconsistencyDataset.owl.xml"
    html_file = "Index of _datasets_InconsistencyDataset.html"

    compare_and_download(rdf_file, html_file)