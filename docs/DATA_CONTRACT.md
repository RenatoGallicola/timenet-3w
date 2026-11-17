# Contratto dei dati

Formato comune tra la parte "Dati e TimeNet" e la parte "Modello e demo".
**Da completare dopo l'esplorazione di 3W** e da approvare da entrambi. Una volta approvato, ogni
modifica va concordata e registrata in `docs/DECISIONS.md`.

## Da decidere
| Voce | Proposta | Stato |
|---|---|---|
| Classi incluse nella valutazione principale | Quelle con abbastanza casi reali e pozzi diversi | da definire dopo i conteggi |
| Classi rare | Training sui simulati, test sui reali, esperimento a parte | da confermare |
| Variabili usate | Le variabili presenti nella maggior parte dei casi reali | da definire |
| Lunghezza della finestra e frequenza | da definire (i dati sono a 1 Hz) | |
| Etichetta di una finestra | da definire (es. classe dominante, fase transitoria o stazionaria) | |
| Split | Per pozzo: nessun pozzo in più di uno split | da confermare |
| Formato dello scambio prima che il connettore sia pronto | File Parquet o NumPy con finestre, etichette, ID del pozzo, provenienza | da definire |
