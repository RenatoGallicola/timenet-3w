# Contratto dei dati

Formato comune tra la parte "Dati e TimeNet" e la parte "Modello e demo".
Approvato il 2026-11-20. Ogni modifica va concordata e registrata in `docs/DECISIONS.md`.

Lo script che lo implementa è `scripts/make_windows.py`: legge i Parquet di 3W (commit
`c86cc7c70063646910be7292de8a54a6fb2a8399`) e scrive le finestre. Quando il connettore è pronto,
lo stesso formato verrà prodotto leggendo da TimeNet.

## Valutazione principale

| Voce | Decisione |
|---|---|
| Casi usati | Solo **reali** (`WELL-*`) |
| Classi | **0, 2, 4, 7, 8, 9**: le uniche con almeno 6 pozzi reali |
| Classi rare (1, 3, 5, 6) | Fuori dalla valutazione principale. Eventuale esperimento a parte: training sui simulati, test sui reali |
| Variabili (6, in quest'ordine) | `P-TPT`, `T-TPT`, `P-ANULAR`, `P-MON-CKP`, `P-JUS-CKGL`, `T-JUS-CKP` |
| Unità | Quelle originali di 3W (Pa, °C), senza normalizzazione. La normalizzazione la fa il modello |
| Finestra | **30 minuti** (1.800 s a 1 Hz) |
| Ricampionamento | Media su blocchi di 10 s → **0,1 Hz, 180 punti** per variabile. Un blocco senza valori resta NaN |
| Passo tra finestre | 15 minuti (900 s) per le normali; **5 minuti** (300 s) per quelle con un evento, che dura solo 1-3 ore |
| Massimo per caso | 20 finestre per caso, scelte a intervalli regolari, per non far pesare troppo i casi lunghi |

## Etichetta di una finestra

Ogni secondo di 3W ha un'etichetta: 0 (normale), `c` (evento `c` stazionario), `100 + c` (evento
`c` transitorio), oppure vuota.

| Etichetta della finestra | Regola |
|---|---|
| Evento `c` | Almeno il **50%** dei secondi ha etichetta `c` o `100 + c` |
| Normale (0) | Almeno il **95%** dei secondi ha etichetta 0 |
| Scartata | Tutti gli altri casi, e le finestre con più del 20% di secondi senza etichetta |

Le finestre "normali" possono venire anche dalla parte normale dei casi con un evento: così la
classe normale copre più pozzi.

Per ogni finestra si salva anche la **fase**: `transient` se la maggioranza dei secondi dell'evento
è transitoria, altrimenti `steady` (per le normali: `normal`). Le metriche si riportano su tutte le
finestre e, a parte, sulle sole finestre `transient`: è la misura dell'allerta precoce. Le classi
3 e 4 di 3W non hanno fase transitoria.

## Variabili mancanti (mai riempite)
- Una variabile è **presente** in una finestra se almeno l'80% dei suoi 180 punti non è NaN.
- I valori mancanti restano NaN nei file. Non si interpolano.
- Nel prompt di OpenTSLM entrano solo le variabili presenti (la maschera è salvata).
- Le finestre con meno di 2 variabili presenti si scartano.

## Split
- **Per pozzo:** nessun pozzo compare in più di uno split.
- Circa 70% / 15% / 15% dei pozzi in train / validation / test, scelto con una ricerca a seme
  fisso: ogni classe ha almeno il 5% delle sue finestre sia in validation sia in test, e le quote
  per classe sono il più vicine possibile al 70/15/15.
- **Da decidere (vedi `docs/STATUS.md`):** usare `splits/wells.json` o lo split del connettore
  della PR #833, salvato in `splits/wells_connector.json` (si passa con `--splits`).
- L'assegnazione dei pozzi è salvata in `splits/wells.json` (nel repo). Il connettore TimeNet
  deve usare la stessa assegnazione; se non è possibile, vale questo file e la differenza va
  dichiarata.

## Formato dello scambio
Nella cartella `data/windows/` (fuori da git):

| File | Contenuto |
|---|---|
| `{train,val,test}.npz` | `X`: float32 `[N, 6, 180]` con NaN per i mancanti; `mask`: bool `[N, 6]` (variabile presente); `y`: int `[N]` (classe 3W: 0, 2, 4, 7, 8, 9) |
| `{train,val,test}.csv` | Una riga per finestra, nello stesso ordine: `window_id`, `instance`, `well`, `label`, `phase`, `start_s`, `event_frac`, `transient_frac` |
| `meta.json` | Commit di 3W, variabili, nomi delle classi, parametri della finestra, conteggi |
