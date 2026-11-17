# Progetto del connettore `petrobras/3w`

**Stato: bozza da approvare.** Secondo la skill `add-dataset-connector` di TimeNet, il codice si
scrive solo dopo l'approvazione di ID, sorgente, tipo di task ed esempio di record.

Basato su: 3W v2.0.0 al commit `c86cc7c70063646910be7292de8a54a6fb2a8399`, l'esplorazione in
`docs/3W_EXPLORATION.md` e i connettori esistenti di TimeNet (soprattutto `physionet/sleep-edfx`,
che come 3W ha registrazioni lunghe con etichette per intervallo).

---

## 1. Identità
| Voce | Proposta |
|---|---|
| ID | `petrobras/three-w`, cartella `datasets/petrobras/three_w/`. TimeNet ricava il nome della cartella dall'ID (minuscolo, trattini → underscore): `petrobras/3w` darebbe una cartella `3w`, che non è un nome di modulo Python importabile |
| Versione | `1.0.0` (versione della conversione TimeNet; la versione di 3W, 2.0.0, va nella descrizione) |
| Licenza | `CC-BY-4.0` |
| Domini | `industrial`, `energy` |
| Tag | `oil-and-gas`, `fault-detection`, `anomaly-detection`, `multivariate`, `petrobras` |
| `source_url` | https://github.com/petrobras/3W |

## 2. Sorgente e download
- Nessuna base esistente fa al caso nostro (non è Hugging Face né PhysioNet): sottoclasse diretta
  di `BaseConnector`.
- I 2.228 file Parquet sono in git senza LFS, quindi ognuno ha un URL fisso al commit:
  `https://raw.githubusercontent.com/petrobras/3W/<commit>/dataset/<classe>/<file>.parquet`.
- **Manifest** `files.csv` nella cartella del connettore: percorso, dimensione e sha256 di ogni
  file, generato una volta da uno script e committato. `download` passa i 2.228 `Artifact` a
  `download_files`, che li scarica in parallelo e verifica ogni sha256.
- `download` fa solo I/O; `convert` legge i file senza rete, e i valori dei segnali si caricano in
  modo pigro (*lazy*), mai tutti in memoria.

## 3. Struttura: un record per caso

| Elemento TimeNet | Da cosa viene in 3W |
|---|---|
| **Record** | Un file Parquet (un caso). Durata = numero di righe, a 1 Hz (verificato: tutti i 2.228 file hanno passo di 1 s esatto) |
| `Record.metadata` | `instance` (nome del file), `well` (es. `WELL-00002`, solo per i reali), `provenance` (`real` / `simulated` / `drawn`), `event_class` (0-9), `start_time` (primo timestamp) |
| **Source** | Gruppi di sensori per posizione fisica (vedi sotto) |
| **Signal** | Una variabile di 3W, con unità di misura. Asse regolare a 1 Hz |
| **Annotazioni sugli intervalli** | Le fasi dell'evento ricavate dalla colonna `class` |
| **Annotazioni sul record** | Provenienza, pozzo, tipo di evento |
| **Task** | Classificazione del tipo di evento e localizzazione della fase transitoria |

### Source (gruppi per posizione)
| Source | Variabili | Unità |
|---|---|---|
| Fondo pozzo (PDG) | `P-PDG`, `T-PDG` | Pa, °C |
| Albero di Natale sottomarino | `P-TPT`, `T-TPT`, `PT-P`, `P-ANULAR` | Pa, °C |
| Valvola di produzione (PCK) | `P-MON-CKP`, `P-JUS-CKP`, `T-MON-CKP`, `T-JUS-CKP`, `ABER-CKP`, `P-MON-SDV-P` | Pa, °C, % |
| Linea del gas lift | `P-MON-CKGL`, `P-JUS-CKGL`, `ABER-CKGL`, `QGL` | Pa, %, m³/s |
| Linea di servizio | `P-JUS-BS`, `QBS` | Pa, m³/s |
| Stati delle valvole | `ESTADO-DHSV`, `ESTADO-M1`, `ESTADO-M2`, `ESTADO-PXO`, `ESTADO-SDV-GL`, `ESTADO-SDV-P`, `ESTADO-W1`, `ESTADO-W2`, `ESTADO-XO` | adimensionale (0 / 0,5 / 1) |
| Stato del pozzo | `state` (codici 0-8) | adimensionale |

### Dati mancanti (regola del repo: mai riempire)
- Una variabile **sempre vuota** in un caso non diventa un segnale di quel record: per quel pozzo
  il sensore non esiste.
- Una variabile **vuota solo in parte** diventa un segnale con
  `MissingValues(reason=NOT_RECORDED)`; i null restano null.
- Una variabile **costante per tutto il caso** (es. `P-PDG` fisso a 0 nell'esempio sotto) resta
  così com'è, con un'annotazione sul segnale `constant_value=true`, perché chi la usa sappia che è
  probabilmente un sensore bloccato. Non la togliamo e non la correggiamo.

### Fasi dell'evento (dalla colonna `class`)
Ogni tratto continuo con la stessa etichetta diventa un'annotazione su intervallo:

| Valore di `class` | Annotazione |
|---|---|
| 0 | `event_phase = normal` |
| 1-9 | `event_phase = steady`, `event_type = <nome dell'evento>` |
| 101-109 | `event_phase = transient`, `event_type = <nome dell'evento>` |
| vuoto | nessuna annotazione (circa il 5% degli istanti) |

## 4. Task

| Task | Uno per | Risposta | Note |
|---|---|---|---|
| `ClassificationTask` "tipo di evento" | Ogni caso (2.228) | Una delle 10 etichette (`Normal operation`, `Abrupt increase of BSW`, ...) | `LabelSet` con le descrizioni di `dataset.ini` |
| `TemporalLocalizationTask` "fase transitoria" | Ogni caso con un evento che ha fase transitoria | Gli intervalli transitori | Modalità `EXHAUSTIVE`: è la base per l'allerta precoce |

Le **finestre** (30-60 minuti per il training) **non** stanno nel connettore: sono una scelta del
nostro progetto, definita in `docs/DATA_CONTRACT.md`. Il connettore resta fedele ai dati originali,
come fanno gli altri connettori di TimeNet.

## 5. Split
- **Casi reali:** split per pozzo (nessun pozzo in più di uno split) con lo `StratifiedSplitter`
  di TimeNet, stratificato per classe e raggruppato per pozzo: train / validation / test.
- **Casi simulati e disegnati:** sempre in train, perché non vengono da pozzi reali.
- **Limite da dichiarare:** le classi 3 e 6 hanno solo 2 pozzi reali, quindi potrebbero mancare
  da validation o test. Lo scriviamo nella scheda del dataset.

## 6. Esempio concreto: `WELL-00002_20131104004101`
Caso reale di **chiusura spuria della DHSV** (classe 2), 12.721 s (3 h 32 min), dal
2013-11-04 00:41:01.

**Record**
```text
metadata = {instance: "WELL-00002_20131104004101", well: "WELL-00002",
            provenance: "real", event_class: 2, start_time: "2013-11-04T00:41:01"}
```

**Segnali** (solo quelli presenti in questo caso)
| Source | Segnale | Unità | Note |
|---|---|---|---|
| Albero di Natale sottomarino | `P-TPT` | Pa | completo |
| Albero di Natale sottomarino | `T-TPT` | °C | completo |
| Albero di Natale sottomarino | `P-ANULAR` | Pa | completo |
| Fondo pozzo (PDG) | `P-PDG` | Pa | costante a 0 → `constant_value=true` |
| Stato del pozzo | `state` | — | sempre 0 |

**Annotazioni sugli intervalli**
| Da (s) | A (s) | Annotazione |
|---|---|---|
| 0 | 3.600 | nessuna (istanti senza etichetta) |
| 3.600 | 7.355 | `event_phase=normal` |
| 7.355 | 12.504 | `event_phase=transient`, `event_type=Spurious closure of DHSV` |
| 12.504 | 12.721 | `event_phase=steady`, `event_type=Spurious closure of DHSV` |

**Task**
- `ClassificationTask`: risposta `Spurious closure of DHSV`.
- `TemporalLocalizationTask`: risposta l'intervallo `[7.355 s, 12.504 s)`.

---

## Punti da decidere insieme
1. **ID:** `petrobras/three-w` (proposta), perché `petrobras/3w` produrrebbe una cartella non
   importabile da Python.
2. **Variabili sempre vuote:** le togliamo dal record (proposta) o le teniamo come segnali tutti
   null?
3. **Codici di `state`:** il significato (0-8) non è scritto nei file di 3W. Proposta: li teniamo
   come codici numerici e cerchiamo la legenda nel toolkit 3W; se non la troviamo, lo dichiariamo.
4. **Split:** casi simulati e disegnati sempre in train (proposta), o anche loro divisi?
5. **Il task di localizzazione** nella prima PR (proposta: sì, perché sostiene l'allerta precoce),
   o in una PR successiva per tenere la prima più piccola?
