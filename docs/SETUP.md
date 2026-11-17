# Setup da zero

Istruzioni per preparare un PC nuovo e cominciare a lavorare sul progetto. Tempo stimato: 45-60
minuti, più l'attesa per l'approvazione di Llama su Hugging Face.

I comandi sono per **Windows (PowerShell)**. Su macOS o Linux funzionano uguali, salvo dove indicato.

---

## 1. Accetta gli inviti GitHub
Sei stata aggiunta come collaboratrice su due repo. Accetta entrambi gli inviti (email di GitHub,
oppure https://github.com/notifications):
- https://github.com/RenatoGallicola/timenet-3w — il progetto
- https://github.com/RenatoGallicola/TimeNet — il fork di TimeNet con il connettore

Senza accettarli non puoi fare `push`.

## 2. Installa gli strumenti
| Strumento | Come | Verifica |
|---|---|---|
| Git | https://git-scm.com/downloads | `git --version` |
| uv | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` (macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh \| sh`) | `uv --version` |
| Python | Non serve installarlo a mano: uv scarica la versione giusta | |
| VS Code + Claude Code | https://code.visualstudio.com, poi l'estensione "Claude Code" | |
| GitHub CLI (consigliato) | https://cli.github.com, poi `gh auth login` | `gh auth status` |

Configura git con il tuo nome (una volta sola):
```powershell
git config --global user.name "Il tuo nome"
git config --global user.email "la-tua-email-github"
```

## 3. Clona il repo del progetto
Scegli una cartella di lavoro, per esempio `Desktop\TimeNet`, e lavora sempre lì dentro:
```powershell
mkdir $HOME\Desktop\TimeNet
cd $HOME\Desktop\TimeNet
git clone https://github.com/RenatoGallicola/timenet-3w.git
```
Il fork di TimeNet per ora non ti serve: lo usa chi lavora sul connettore. Quando il connettore è
pronto, ti diremo come installarlo.

## 4. Crea gli account (tutti gratuiti)

### Hugging Face
1. Registrati su https://huggingface.co
2. Chiedi l'accesso a https://huggingface.co/meta-llama/Llama-3.2-1B (pulsante in cima alla pagina).
   L'approvazione può richiedere qualche ora: fallo per primo.
3. Accetta la licenza di Gemma su https://huggingface.co/google/gemma-3-1b-pt
4. Crea un token: Settings → Access Tokens → New token, tipo **Read**.

### Modal (GPU per il training)
1. Registrati su https://modal.com con il tuo account GitHub.
2. Aggiungi la carta (Modal la richiede), poi **subito**: Settings → Usage & Billing → imposta il
   **spend limit a 0** (o al minimo consentito). Così usi solo i $30 gratuiti al mese e non paghi mai.
3. Usa un tuo workspace: i $30 sono per workspace, quindi non entrare in quello dell'altro membro.
4. Il collegamento da terminale (`modal token new`) lo fai dopo, dentro l'ambiente del progetto:
   te lo guiderà Claude.

### Kaggle (GPU di riserva)
1. Registrati su https://www.kaggle.com
2. Settings → verifica il numero di telefono (senza non si usano le GPU).

## 5. Metti i token nel file `.env`
Nel repo c'è `.env.example`. Copialo in `.env` e compila i valori:
```powershell
cd $HOME\Desktop\TimeNet\timenet-3w
copy .env.example .env
notepad .env
```
`.env` è escluso da git: **non va mai committato**, e i token non vanno scritti in nessun altro file.

## 6. Scarica 3W allo stesso commit
Tutto il team usa 3W v2.0.0 al commit `c86cc7c70063646910be7292de8a54a6fb2a8399`. Scaricalo in una
cartella `data` accanto al repo, fuori da `timenet-3w`. Sono circa 1,9 GB:
```powershell
cd $HOME\Desktop\TimeNet
mkdir data
cd data
git -c core.autocrlf=false clone --filter=blob:none --sparse --no-checkout https://github.com/petrobras/3W.git 3w
cd 3w
git sparse-checkout set dataset
git checkout c86cc7c70063646910be7292de8a54a6fb2a8399
```
Verifica: nella cartella `dataset` devono esserci 10 sottocartelle (0-9) e 2.228 file `.parquet`.
Poi scrivi il percorso in `.env` alla voce `THREEW_DATASET_DIR`, per esempio
`C:\Users\<tu>\Desktop\TimeNet\data\3w\dataset`.

## 7. Avvia Claude Code e dagli il primo prompt
Apri la cartella `timenet-3w` in VS Code, apri Claude Code e imposta l'effort **medium**. Primo prompt:

> Leggi CLAUDE.md, docs/PLAN.md, docs/STATUS.md, docs/DATA_CONTRACT.md e docs/3W_EXPLORATION.md.
> Il mio ruolo è "Modello e demo". Verifica che il setup di docs/SETUP.md sia completo sul mio PC,
> poi aiutami con i miei compiti di oggi: collegare Modal, clonare OpenTSLM e far girare la demo
> HAR di OpenTSLM su una GPU A100 di Modal. Aggiorna docs/STATUS.md alla fine.

Claude troverà in CLAUDE.md tutto il contesto: scadenze, vincoli, decisioni prese e fatti già
verificati.

## 8. Le regole di ogni sessione
- **All'inizio:** `git pull`, poi lettura di `docs/STATUS.md`.
- **Durante:** commit piccoli e frequenti, messaggi nel formato `tipo: descrizione`
  (es. `feat: add 3W window loader`).
- **Alla fine:** aggiorna `docs/STATUS.md`, poi `git push`.
- **Decisioni:** annotale in `docs/DECISIONS.md` con data e motivo.
- **Prima di toccare file dell'altra parte:** scrivilo in `docs/STATUS.md`, per evitare conflitti.
