Cel projektu: Zbudowanie dwóch przepływów opartych na LLM.
Wymagana wersja Pythona >= 3.11

#### Sposób instalacji:

# Utwórz wirtualne środowisko w folderze projektu 
python -m venv .venv

# Aktywuj środowisko: 

Windows PowerShell:
.\.venv\Scripts\Activate.ps1

Linux lub macOS:
source .venv/bin/activate

# Zainstaluj wymagane pakiety:
pip install -r requirements-lock.txt

Aby wybrać tryb local / api:
W pliku .env ustaw zmienną LLM_MODE na local / api

Uruchomienie testów:

pytest

Znane ograniczenia:

Uruchomienie trybu lokalnego na samym procesorze może skutkować wolnym generowaniem odpowiedzi.
