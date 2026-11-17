# Piano

Aggiornato al 17 novembre 2026. Si punta a fare **tutto**, ma nell'ordine di priorità qui sotto:
in ogni momento deve esistere un progetto consegnabile.

## Ruoli
| Ruolo | Responsabilità | Chi |
|---|---|---|
| **Dati e TimeNet** | Connettore 3W, test, CI, PR e rapporti con i maintainer; contratto dei dati e split; valutazione, baseline e statistiche | RenatoGallicola |
| **Modello e demo** | Collegamento TimeNet → OpenTSLM, finestre e spiegazioni, training su Modal, demo | ariannabalducci |
| **Entrambi** | Post su X e LinkedIn, video, consegna | |

Dopo il connettore (da giovedì), chi fa Dati e TimeNet prende anche blocchi interi della parte
modelli, scelti tra: baseline forti, test di perturbazione, secondo training (Flamingo o
sim-to-real), verifica della qualità delle spiegazioni. La valutazione finale sui pozzi di test resta
a Dati e TimeNet, per tenerla indipendente da chi addestra. Prima di prendere un blocco, scriverlo in
`docs/STATUS.md`.

Chi parte da un PC nuovo segue `docs/SETUP.md`.

Le due parti procedono in parallelo. Il punto di contatto è `docs/DATA_CONTRACT.md`: chi lavora
sul modello prototipa leggendo i Parquet di 3W direttamente e passa al loader TimeNet quando il
connettore è pronto.

## Ordine di priorità
1. **Indispensabile**
   - PR del connettore 3W: classificazione degli eventi, intervalli transitori come annotazioni,
     provenienza reale/simulata/disegnata, ID del pozzo, split per pozzo.
   - Un modello addestrato (OpenTSLM SoftPrompt, Llama 3.2 1B) sulle classi con abbastanza casi reali.
   - Test su pozzi esclusi dal training.
   - Baseline: XGBoost su feature e un classificatore che vede solo media e deviazione standard.
   - Spiegazioni costruite dalle feature calcolate, con verifica automatica.
   - Un test di perturbazione (segnale mescolato nel tempo).
   - Demo con esempi già calcolati, README, post, consegna.
2. **Poi**
   - Flamingo e confronto tra le due architetture.
   - Training sui simulati e test sui reali per le classi rare.
   - Tutti i test di perturbazione (scambio tra classi, statistiche rimosse).
   - Misura di quanto in anticipo viene riconosciuto l'evento.
   - Baseline con un LLM grande non addestrato.
3. **Se avanza tempo**
   - Localizzazione temporale come task separato.
   - Spiegazioni riscritte da un modello open.
   - Demo live su GPU, video curato.

## Calendario
| Giorno | Dati e TimeNet | Modello e demo |
|---|---|---|
| Mar 17 | Ambiente, download di 3W da un commit fisso, conteggi, progetto del connettore | Account Modal, OpenTSLM funzionante su A100, struttura del collegamento |
| Mer 18 | Connettore, test, CI → **PR aperta** | Finestre, feature, spiegazioni da template, baseline |
| Gio 19 | Correzioni della PR, script di valutazione | **Primo training completo** |
| Ven 20 | Analisi dei risultati, grafici | Secondo training o confronto, perturbazioni |
| Sab 21 | README e pagina del progetto, post | Demo su Hugging Face, video |
| Dom 22 | **Consegna entro le 20:00** | Margine |

**Punto di controllo, giovedì sera:** si decide l'ordine del lavoro dei giorni finali in base ai
risultati del primo training.

## Post social
1. Oggi: partecipiamo e portiamo 3W in TimeNet.
2. All'apertura della PR.
3. Con i risultati e la demo.
