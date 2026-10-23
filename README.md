Cel projektu: Zbudowanie dwóch przepływów opartych na LLM.
Wymagana wersja Pythona >= 3.11

#### Sposób instalacji:

# Utwórz wirtualne środowisko w folderze projektu 
python -m venv .venv

# Aktywuj środowisko: 

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux lub macOS:
source .venv/bin/activate

# Zainstaluj wymagane pakiety:
pip install -r requirements-lock.txt

Aby wybrać tryb local / api:
W pliku .env ustaw zmienną LLM_MODE na local / api

Uruchomienie testów:

pytest

Znane ograniczenia:

Uruchomienie trybu lokalnego na samym procesorze może skutkować wolnym generowaniem odpowiedzi.

Dlaczego liczby tokenów nie można wiarygodnie oszacować liczbą słów:
Tokenizery dzielą tekst na fragmenty zależnie od specyfiki danego modelu i języka. Dla jednego modelu jedno słowo może posiadać 3 tokeny a dla drugiego 5.

Potwierdzenie rozwoju repozytorium:
To repozytorium będzie używane i rozwijane w ramach laboratorium.