# Esplorazione di 3W

Eseguita il 2026-11-17 con `scripts/explore_3w.py` su 3W v2.0.0, commit
`c86cc7c70063646910be7292de8a54a6fb2a8399` (14 novembre 2026), cartella `dataset/`:
2.228 file Parquet, 1,9 GB.

## Struttura dei file
- Una cartella per classe (0-9), un file Parquet per caso.
- Il prefisso del nome dà la provenienza: `WELL-xxxxx_` (reale, con ID del pozzo), `SIMULATED_`, `DRAWN_`.
- Colonne: `timestamp`, 27 variabili, `class` (etichetta per istante), `state` (stato operativo del pozzo).
- Frequenza 1 Hz. Etichetta transitoria = classe + 100.

## Casi per classe

| Classe | Evento | Reali | Pozzi reali | Simulati | Disegnati |
|---|---|---|---|---|---|
| 0 | Normale | 594 | 9 | 0 | 0 |
| 1 | Aumento improvviso di BSW | 4 | 3 | 114 | 10 |
| 2 | Chiusura spuria della DHSV | 22 | 7 | 16 | 0 |
| 3 | Slugging severo | 32 | 2 | 74 | 0 |
| 4 | Instabilità di flusso | 343 | 7 | 0 | 0 |
| 5 | Perdita rapida di produttività | 11 | 3 | 439 | 0 |
| 6 | Restrizione rapida della PCK | 6 | 2 | 215 | 0 |
| 7 | Incrostazioni nella PCK | 36 | 6 | 0 | 10 |
| 8 | Idrati in linea di produzione | 14 | 9 | 81 | 0 |
| 9 | Idrati in linea di servizio | 57 | 15 | 150 | 0 |

Totale: 1.119 reali da 40 pozzi, 1.089 simulati, 20 disegnati.

**Conseguenza per lo split:** le classi 0, 2, 4, 7, 8, 9 hanno almeno 6 pozzi reali, quindi si può
valutare per pozzo. Le classi 1, 3, 5, 6 hanno 2-3 pozzi reali: vanno trattate come esperimento
sim-to-real (training sui simulati, test sui pochi reali).

## Variabili davvero utilizzabili nei casi reali
Quota dei casi reali in cui la variabile è presente per più di metà del tempo e non è costante:

| Variabile | Significato | Quota |
|---|---|---|
| T-TPT | Temperatura all'albero di Natale sottomarino (TPT) | 91% |
| P-TPT | Pressione al TPT | 82% |
| P-ANULAR | Pressione nell'anulus del pozzo | 82% |
| P-MON-CKP | Pressione a monte della valvola di produzione (PCK) | 65% |
| P-JUS-CKGL | Pressione a valle della valvola del gas lift | 60% |
| T-JUS-CKP | Temperatura a valle della PCK | 50% |
| T-PDG | Temperatura di fondo pozzo (PDG) | 25% |
| ABER-CKP | Apertura della PCK | 25% |
| P-PDG | Pressione di fondo pozzo (PDG) | 16% |
| QGL | Portata del gas lift | 13% |
| tutte le altre | | < 5% |

Le variabili `ESTADO-*` sono stati di valvole (0, 0,5, 1): risultano quasi sempre costanti, il che
per uno stato è normale, ma vuol dire che portano informazione solo quando cambiano (per esempio la
chiusura della DHSV).

**Conseguenza:** il nucleo è di circa 6 variabili. Le altre sono spesso mancanti e vanno dichiarate
come tali nel formato TimeNet, senza riempirle.

## Durata dei casi (in secondi, cioè righe a 1 Hz)
| Provenienza | Minimo | Mediana | Massimo |
|---|---|---|---|
| Reali | 5.215 (1,4 h) | 21.354 (5,9 h) | 768.314 (8,9 giorni) |
| Simulati | 12.999 | 29.299 (8,1 h) | 88.800 |
| Disegnati | 14.400 | 86.400 | 345.601 |

**Conseguenza:** i casi sono troppo lunghi per il modello: servono finestre (per esempio 30-60
minuti) ricampionate a frequenza più bassa.

## Istanti per tipo di etichetta
| Provenienza | Normale | Transitorio | Stazionario |
|---|---|---|---|
| Reali | 14,6 M | 11,0 M | 3,3 M |
| Simulati | 2,4 M | 12,7 M | 25,7 M |
| Disegnati | 0,4 M | 2,5 M | 0,1 M |

Nei casi reali la fase transitoria è molto rappresentata: c'è materiale per l'allerta precoce.
