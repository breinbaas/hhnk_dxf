# DXF naar STIX

Dit script converteert DXF bestanden die binnen HHNK gemaakt worden naar Deltares DStability (stix) bestanden.

## DXF eisen

Het script werkt met het DXF format dat HHNK gedefinieerd heeft. Bij dit format bestaat het maaiveld uit een lwpolyline.
De linker- en rechter begrenzingen van de geometrie zijn gedefinieerd als een enkele lwpolyline van de onderzijde tot
de bovenzijde (en bevat dus **niet** de snijpunten met de tussenliggende lagen).

De overige lijnen die de grondlagen weergeven zijn losse lijnen.

Het script werkt met deze aanname en bij wijzigingen in dit format is niet gegarandeerd dat het script goede resultaten oplevert.

## Installatie

Op verzoek van de opdrachtgever wordt gebruik gemaakt van de main branch van de d-geolib code. Dit in verband met het feit dat
het beschikbare python package nog niet geupdate is voor de huidige versie van DStability (v2024.02). Dit houdt in dat de
afhankelijkheid van geolib niet als package meegenomen kan worden via pip of in de requirements.txt. De workaround voor het 
gebruik van de main branch van geolib is als volgt;

* checkout de main versie van geolib bv via ```git clone git@github.com:Deltares/GEOLib.git```
* onthoud naar welk pad de code gedownload is, bv ```C:\MyCode\geolib```
* maak binnen Windows een PYTHONPATH variabele aan (Windows toets | Edit the System Variables | Environment Variables | NEW) met als naam ```PYTHONPATH``` en als waarde het pad, bv ```C:\MyCode\geolib```

**let op** als er al een PYTHONPATH variabele aan is gemaakt dan kun je met de puntkomma een extra pad toevoegen bv ```C:\MyCode\anderpad;C:\MyCode\geolib```
**let op** als er een IDE open stond tijdens dit proces zijn de wijziging pas werkzaam als de IDE herstart wordt

Download de code van het script via;
* git clone 

