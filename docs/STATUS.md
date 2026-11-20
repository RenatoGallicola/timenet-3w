# Stato

Aggiornare a ogni fine sessione. Le voci più recenti in alto.

## Adesso (venerdì 20 novembre, sera)
Mancano circa 2 giorni: consegna entro **domenica 22 alle 20:00** (scadenza vera: lunedì 23 alle
08:59 ora italiana).

> **Per RenatoGallicola:** il contratto dei dati, l'estrazione delle finestre e le baseline sono
> **già fatti** (vedi "Fatto" sotto): non rifarli. Oggi abbiamo anche scritto in parallelo lo stesso
> connettore senza saperlo: quello di ariannabalducci resta solo in locale e non verrà pushato. Vale
> la PR #833. Prima di iniziare un blocco, scriverlo qui.

| Chi | Sta facendo | Blocchi |
|---|---|---|
| RenatoGallicola (Dati e TimeNet) | Decidere con ariannabalducci lo split (sotto) e i ritocchi alla PR #833; poi, a scelta: baseline con i casi simulati nel training, confronto split casuale contro split per pozzo, test di perturbazione | Nessuno |
| ariannabalducci (Modello e demo) | Collegamento delle finestre a OpenTSLM; training su Modal | Nessuno: accesso a Llama 3.2 1B già concesso |

## Da decidere subito (insieme)
1. **Quale split usare nel training.** Oggi esistono due split per pozzo diversi:
   - `splits/wells.json` (scelto da `scripts/make_windows.py`, bilanciato sulle finestre);
   - quello del connettore della PR #833 (`StratifiedSplitter` 3/1/1, bilanciato sui casi).

   **Proposta: usare quello del connettore**, così progetto e PR usano gli stessi dati. È stato
   verificato con `make_windows.py --splits` e funziona: ogni classe principale ha finestre in val e
   in test.

   | Split | Finestre train / val / test | Pozzi train / val / test | Baseline `shape` test | `mean_std` test |
   |---|---|---|---|---|
   | `splits/wells.json` | 10.045 / 6.344 / 2.348 | 27 / 6 / 6 | 0,35 | 0,32 |
   | connettore PR #833 | 12.827 / 2.690 / 3.220 | 24 / 7 / 8 | 0,39 | 0,26 |

   Con lo split del connettore il train ha poche finestre delle classi 7 e 9 (180 e 70). La
   cross-validation per pozzo non dipende dallo split: 0,38-0,40.
2. **Ritocchi alla PR #833**, presi dal connettore scritto in parallelo:
   - **Fuso orario:** `_start_time` assume UTC (`datetime.fromtimestamp(..., tz=UTC)`), ma i
     timestamp di 3W non hanno fuso orario e il rilascio non lo dichiara. Non è un dato verificato.
     Proposta: niente `start_time` assoluto e l'inizio come testo in `Record.metadata`, oppure una
     fonte che dica il fuso.
   - **Scheda del dataset:** mancano `license_url` (la licenza CC BY dei dati è in
     `dataset/LICENSE-CC-BY` del rilascio; `LICENSE.md` è la licenza Apache del codice) e
     `citation`. Per la 2.0.0, `CITATION.md` di 3W indica: Vaz Vargas, R.E., de Melo Junior, A.J.,
     de Campos Lima, C.B. et al. *3W Dataset 2.0.0: a realistic and public dataset with rare
     undesirable real events in oil wells.* Sci Data 13, 949 (2026),
     https://doi.org/10.1038/s41597-026-07225-z. C'è anche il campo `source_revision` per il commit.
   - **Facoltativo:** un `README.md` nella cartella del connettore con le decisioni, come
     `physionet/sleep_edfx`.

## Fatto
- 2026-11-20 (ariannabalducci): **contratto dei dati** approvato e implementato
  (`docs/DATA_CONTRACT.md`, `scripts/make_windows.py`): 18.737 finestre di 30 minuti da 39 pozzi
  reali, in circa 40 secondi. Il pozzo `WELL-00008` non ha finestre delle classi principali.
- 2026-11-20 (ariannabalducci): **baseline XGBoost** (`scripts/baselines.py`, risultati in
  `results/baselines.json`). Macro-F1 su 6 classi, pozzi mai visti: feature di forma 0,35 in test,
  0,39 ± 0,13 in cross-validation per pozzo; solo media e deviazione standard 0,32 in test, 0,40 ±
  0,14 in cross-validation. Due risultati da raccontare:
  1. ogni pozzo ha sensori diversi: senza correzioni il modello impara quali sensori ci sono, e su
     pozzi nuovi riconosce 0 casi di instabilità di flusso (con i sensori nascosti a caso: F1 0,6);
  2. media e deviazione standard bastano quasi quanto le feature di forma: OpenTSLM scrive proprio
     quei numeri nel prompt, quindi i test di perturbazione sono indispensabili.
- 2026-11-20 (ariannabalducci): guardate le PR dell'hackathon su TimeNet. Nessun altro team usa 3W.
  Un altro team ha preso le turbine eoliche CARE (PR #792). Il maintainer TimoStoff fa revisioni
  dettagliate (PR #788: tipi con nome, validazione più stretta). Alcune PR aperte dai maintainer
  cambiano l'API dei connettori (#821, #822, #827): la PR #833 potrebbe richiedere un rebase.
- 2026-11-20: **PR in bozza aperta**: https://github.com/OpenTSLM/TimeNet/pull/833 (2 commit;
  allineata allo stile dei connettori esistenti). Etichetta `area:connectors` da aggiungere ai
  maintainer: noi non abbiamo i permessi.
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
- Decidere lo split e i ritocchi alla PR (sopra).
- Segnare la PR #833 come "Ready for review" quando il team è soddisfatto.
- Primo post su X e LinkedIn.
- Verificare se è arrivato il voucher Nebius.
