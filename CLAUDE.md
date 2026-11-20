# CLAUDE.md

Contesto condiviso per le sessioni Claude Code di entrambi i membri del team.
Rispondi sempre in italiano. Codice, commenti e README pubblici in inglese.

## Prima di iniziare ogni sessione
1. `git pull` su questo repo (e sul fork di TimeNet, se ci lavori).
2. Leggi `docs/STATUS.md` per sapere chi sta facendo cosa.
3. Controlla `docs/DECISIONS.md` prima di proporre di cambiare qualcosa di già deciso.

## Alla fine di ogni sessione
- Aggiorna `docs/STATUS.md` (cosa è fatto, cosa è in corso, cosa blocca).
- Commit piccoli e frequenti, poi `git push`.
- **Nessuna attribuzione a Claude:** niente righe `Co-Authored-By: Claude` nei commit e niente
  "Generated with Claude Code" nelle PR. Commit e PR risultano solo di chi li fa.
- Se prendi una decisione, aggiungila a `docs/DECISIONS.md` con data e motivo.

## Il progetto in breve
- **Hackathon:** TimeNet Online Hackathon, Round 1 (Aionic Labs + ETH Zurich Agentic Systems Lab).
  Scadenza: **22 novembre 2026, 23:59 PST = 23 novembre, 08:59 ora italiana**. Obiettivo interno:
  consegnare **domenica 22 entro le 20:00**.
- **Caso d'uso:** riconoscere e spiegare gli eventi anomali nei pozzi petroliferi, con il dataset
  [Petrobras 3W](https://github.com/petrobras/3W) v2.0.0.
- **Criteri di giudizio:** caso d'uso e impatto; esecuzione tecnica; dati e TimeNet; risultati e
  valutazione; presentazione e social (contano i post su X e LinkedIn).
- **Consegne:** link alla PR su OpenTSLM/TimeNet che aggiunge il dataset; link a demo o sito; questo
  repo; post pubblici su X e LinkedIn con screenshot delle metriche; post su Discord; form di consegna.

Piano completo: `docs/PLAN.md`.

## Repository
| Repo | Uso |
|---|---|
| https://github.com/RenatoGallicola/timenet-3w | Questo repo: collegamento TimeNet → OpenTSLM, training, valutazione, demo, documenti |
| https://github.com/RenatoGallicola/TimeNet | Fork di OpenTSLM/TimeNet: solo il connettore 3W, da cui parte la PR |

## Vincoli
- **Spesa zero.** Solo risorse gratuite: Modal ($30 al mese per workspace, con tetto di spesa a 0),
  Kaggle, Hugging Face ZeroGPU. Nessuna API a pagamento. Nebius solo se arriva il voucher e il team
  accetta la ricarica obbligatoria di 25 USD.
- **OpenTSLM usa bf16:** il training va fatto su A100/H100 (Modal). Le GPU di Kaggle (T4/P100) non
  supportano bf16; vanno bene per generare testo con modelli open.
- **PC locale:** Windows, nessuna GPU NVIDIA, 16 GB di RAM, circa 30 GB liberi, niente WSL. Si usa
  uv direttamente su Windows.
- **Segreti:** token e chiavi (Hugging Face, Modal) solo in `.env`, mai nel repo.

## Regole del repo TimeNet (per il connettore)
- Seguire `AGENTS.md` e la skill `.agents/skills/add-dataset-connector` del repo TimeNet.
- Il progetto del connettore va approvato dal team **prima** di scrivere codice.
- Ogni file sorgente fissato a un commit, con sha256. Dati mancanti dichiarati, **mai riempiti**.
- Commit nel formato Conventional Commits; mai `--no-verify`.

## Fatti già verificati (non ricontrollarli)
- 3W v2.0.0: licenza dati CC BY 4.0; 2.228 file Parquet (1,87 GB) in git senza LFS; 1.119 casi reali
  da 40 pozzi, 1.089 simulati, 20 disegnati a mano; 27 variabili; 10 classi; l'etichetta transitoria
  è classe + 100. Casi reali per classe: normale 594, aumento BSW 4, chiusura spuria DHSV 22,
  slugging severo 32, instabilità di flusso 343, perdita rapida di produttività 11, restrizione
  rapida PCK 6, incrostazioni PCK 36, idrati in linea di produzione 14, idrati in linea di servizio 57.
- OpenTSLM non ha un loader per TimeNet: va scritto. `OpenTSLMSP` fissa bf16 e attention eager;
  LoRA r=16 nelle fasi con rationale. Checkpoint pronti: `OpenTSLM/{llama-3.2-1b,...}-{tsqa,m4,har,sleep,ecg}-{sp,flamingo}`.
- OpenTSLM scrive media e deviazione standard di ogni serie nel prompt: il modello potrebbe usare
  quelle statistiche invece del segnale. Va testato con perturbazioni.
- Paper di riferimento su 3W: arXiv 2608.04041 (LLM da 397B con metriche calcolate, 35,1% top-1 su
  9 classi; protocollo diverso dal nostro).
