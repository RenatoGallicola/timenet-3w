# Decisioni

Una riga per decisione: data, decisione, motivo. Le più recenti in fondo.

| Data | Decisione | Motivo |
|---|---|---|
| 2026-11-11 | Caso d'uso: eventi anomali nei pozzi petroliferi con Petrobras 3W | Primo per 2 revisori indipendenti su 3, secondo per il terzo, su 17 candidati. Dominio "industrial" di TimeNet ancora vuoto, allineato con gli esempi della challenge e con i settori di Aionic Labs, poca concorrenza |
| 2026-11-11 | Scartati Parkinson (Daphnet), WESAD, apnea da ECG, turbine eoliche | Daphnet: solo 29,5 minuti di freezing e domande già nella PR #720. WESAD: nessuna licenza utilizzabile. Apnea: ID dei pazienti mancanti, dominio affollato. Eolico: sensori anonimi, circa 45 guasti |
| 2026-11-12 | Campo "previous work" della registrazione lasciato vuoto | Si parte da un progetto nuovo; evitare ambiguità con repo precedenti non finiti |
| 2026-11-17 | Si punta a fare tutto il piano originale, in ordine di priorità | Scelta del team |
| 2026-11-17 | Training su Modal (gratis, con tetto di spesa a 0); Nebius opzionale | Nebius richiede una ricarica di 25 USD per il voucher; il team vuole spesa zero |
| 2026-11-17 | Repo pubblico `timenet-3w` sul GitHub di uno dei membri; nome da prodotto deciso più avanti | Sviluppo in pubblico; nome provvisorio descrittivo |
| 2026-11-17 | Sviluppo locale su Windows con uv, senza WSL | WSL non installato |
| 2026-11-17 | 3W fissato al commit `c86cc7c70063646910be7292de8a54a6fb2a8399` (v2.0.0) | Ultimo commit al momento del download; tutto il team usa gli stessi dati |
| 2026-11-17 | Ruoli: RenatoGallicola su Dati e TimeNet, ariannabalducci su Modello e demo | Le due parti procedono in parallelo; chi finisce il connettore prende poi blocchi della parte modelli |
| 2026-11-17 | Non si aspetta il merge della PR | Per la consegna basta il link; il progetto installa TimeNet dal fork |
| 2026-11-20 | Connettore con 6 tipi di segnale per grandezza fisica e `TimeNetConnectorError` | Stile dei connettori esistenti di TimeNet |
| 2026-11-20 | Nessuna attribuzione a Claude in commit e PR | Richiesta del team |
