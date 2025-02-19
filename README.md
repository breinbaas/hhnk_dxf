# DXF naar STIX

Dit script converteert DXF bestanden die binnen HHNK gemaakt worden naar Deltares DStability (stix) bestanden.

* 13/02/2025 het script werkt voor DXF tekeningen waarbij aan de onderstaande eisen wordt voldaan, er wordt nog onderzocht hoe het script werkt met polygon DXF tekeningen

## DXF eisen

Het script werkt met het DXF format dat HHNK gedefinieerd heeft. Bij dit format bestaat het maaiveld uit een lwpolyline.
De linker- en rechter begrenzingen van de geometrie zijn gedefinieerd als een enkele lwpolyline van de onderzijde tot
de bovenzijde (en bevat dus **niet** de snijpunten met de tussenliggende lagen).

De overige lijnen die de grondlagen weergeven zijn losse lijnen. Alle punten van de losse lijnen die het maaiveld snijden zijn ook
onderdeel van het maaiveld. 

Het script werkt met deze aanname en bij wijzigingen in dit format is niet gegarandeerd dat het script goede resultaten oplevert.

## Installatie

Download de code van het script via;
* git clone ```git@github.com:breinbaas/hhnk_dxf.git```
* ga naar de directory met de code en maak een virtuele omgeving aan; ```python -m venv .venv```
* installeer de benodigde packages ```python -m pip install -r requirements.txt```

## Script uitvoeren
* activeer de virtuele omgeving ```.venv/scripts/activate```
* plaats de DXF bestanden in de de ```/data``` directory 
* run het script ```python main.py```
* in de data directory worden stix bestanden gegenereerd met dezelfde naam als het dxf bestand maar met ```stix``` als extensie