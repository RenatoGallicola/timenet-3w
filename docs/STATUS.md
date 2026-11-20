# Stato

Aggiornare a ogni fine sessione. Le voci più recenti in alto.

## Adesso (venerdì 20 novembre)
Mancano circa 2 giorni e mezzo: consegna entro **domenica 22 alle 20:00** (scadenza vera: lunedì
23 alle 08:59 ora italiana). Siamo in ritardo di circa un giorno e mezzo sul piano.

| Chi | Sta facendo | Blocchi |
|---|---|---|
| RenatoGallicola (Dati e TimeNet) | Aprire la PR del connettore; scrivere il contratto dei dati e l'estrazione delle finestre da 3W, così il training può partire | Ok finale per aprire la PR |
| ariannabalducci (Modello e demo) | Setup con `docs/SETUP.md`; Modal; OpenTSLM che gira su una GPU A100 di Modal con la demo HAR | Approvazione di Llama 3.2 1B su Hugging Face: chiederla subito |

## Fatto
- 2026-11-17: **connettore 3W completato** nel fork, branch `feature/petrobras-three-w`
  (`datasets/petrobras/three_w/`). ID `petrobras/three-w`. Verificato: ruff, ty (anche Python 3.11),
  11 test, controlli di tutti i connettori, costruzione completa (18 min, 1,1 GB: 2.228 record, 2.228
  task di classificazione, 1.142 di localizzazione). Il dataset costruito è nella cartella
  `registry/` accanto ai repo, sul PC di RenatoGallicola.
- 2026-11-17: ruoli confermati; `docs/SETUP.md` e `.env.example`; ariannabalducci collaboratrice su
  entrambi i repo (invito accettato).
- 2026-11-17: ambiente TimeNet funzionante; 3W scaricato al commit `c86cc7c7` ed esplorato
  (`docs/3W_EXPLORATION.md`); progetto del connettore approvato (`docs/CONNECTOR_DESIGN.md`).
- 2026-11-12: registrazione all'hackathon con il pitch su 3W.

## Da fare a breve
- PR in bozza verso OpenTSLM/TimeNet (nessun altro team ha aperto PR su 3W al 20 novembre).
- Contratto dei dati (`docs/DATA_CONTRACT.md`) e finestre pronte per il training.
- Primo post su X e LinkedIn.
- Verificare se è arrivato il voucher Nebius.
