# Handoff — Meetily diarization: test on-device della build installata

> **Come usare questo file:** è committato su `enhance/diarization`. Sul Mac fai
> `git checkout enhance/diarization && git pull`, poi apri questo file dalla root del repo.
> Al termine dei test, questo file può essere cancellato (non è codice di produzione).

> **Scopo di questa sessione (nuovo device):** riprendere il lavoro su **macOS**, sullo stesso repo `meetily`, per **testare direttamente sul Mac dove è installata la build dell'app** perché il bottone "Identify speakers / enhance diarization" restituisce **"Failed to start speaker identification"**. Diagnosticare la causa reale e correggere.

## Contesto essenziale (chi/dove)
- Repo: fork `cookkie03/meetily` (branch di lavoro **`enhance/diarization`**). Il device precedente era **Linux/Debian** (dev), questo nuovo device è il **Mac** dove gira l'app installata.
- Feature: diarization timbrica pyannote ONNX post-registrazione (Speaker N rinominabili). Piano e stato completi nella memoria Claude Code: `~/.claude/projects/-home-luca-meetily/memory/meetily-diarization-plan.md` e `MEMORY.md`. **Leggile se disponibili su questo device** (la memoria è per-device: se manca, fai riferimento a questo doc e ai commit).
- Ruolo dell'utente: lui è il boss; l'assistente è **project manager** e delega l'operativo a `cdx` (o `agy`). Mantieni questo modello se non diversamente indicato.

## Il problema
Premendo il bottone diarization in un meeting salvato → toast **"Failed to start speaker identification"**.
- Quel toast è **esattamente** `frontend/src/components/MeetingDetails/TranscriptButtonGroup.tsx:167`, dentro il `catch` di `await invoke('run_diarization', { meeting_id })` (riga 161).
- Il comando backend `run_diarization` (`frontend/src-tauri/src/diarization/commands.rs:300`) fa `tokio::spawn` e **ritorna sempre `Ok(())`**. Quindi se `invoke` viene **rifiutato**, la causa NON è la logica interna: è che **quel comando non esiste / non è invocabile nel binario in esecuzione**, oppure un errore di deserializzazione argomenti.

## Diagnosi già stabilita (con prove) sul device precedente
1. Il codice diarization è **solo** su `enhance/diarization` (commit `cecda18`, `4dd7feb`, `ddaba5b`). **NON è in `main`.**
2. `git show main:frontend/src-tauri/src/lib.rs | grep -c run_diarization` → **0**. Main non ha né il comando né il bottone.
3. **Entrambe le Release** GitHub sono state buildate da **`main`** via `workflow_dispatch` (`gh run list --workflow=release.yml` → branch `main`, ultima OK 2026-06-30). → **nessuna release ufficiale contiene la feature.**
4. Conclusione ponte: se l'utente vede il bottone MA `invoke` fallisce, sta girando una build **disallineata** (frontend con bottone, backend senza comando) — plausibilmente una build locale vecchia/parziale da `clean_build`, o comunque non una build pulita di `enhance/diarization`.

**Contraddizione da risolvere on-device:** un build pulito di `enhance/diarization` avrebbe SIA il bottone SIA il comando → `invoke` non dovrebbe fallire. Quindi o (a) la build installata è disallineata, o (b) c'è un vero bug (argomenti/registrazione) da vedere solo a runtime.

## Test decisivo da eseguire SUBITO sul Mac (red/green)
DevTools in build di produzione è disabilitato (Cmd+Shift+I non fa nulla — normale). Non serve. Controllare cosa contiene il binario installato:

```bash
APP=$(ls -d /Applications/*eetily*.app ~/Applications/*eetily*.app 2>/dev/null | head -1)
echo "App: $APP"
BIN="$APP/Contents/MacOS/"$(ls "$APP/Contents/MacOS/" 2>/dev/null | head -1)
echo "Binario: $BIN"
echo -n "run_diarization nel binario Rust: "; strings "$BIN" 2>/dev/null | grep -c run_diarization
echo -n "run_diarization nel frontend JS: "; grep -rl run_diarization "$APP/Contents/Resources/" 2>/dev/null | head
```

Interpretazione:
- **Rust = 0** → comando NON nel binario installato. Nessun bug di codice: la build è disallineata / la feature non è mai stata spedita. **Fix = buildare/installare da `enhance/diarization`** (vedi sotto). Questa è l'ipotesi più probabile.
- **Rust > 0** → il comando c'è → vero bug runtime. Prossimo passo: catturare l'errore reale via log da terminale (vedi sotto) e correggere nel codice (sospetto n.1 = deserializzazione argomenti; convenzione del repo è **snake_case**, es. `mic_device_name`, quindi `{ meeting_id }` dovrebbe essere giusto — verificare comunque).

### Se serve l'errore runtime esatto (caso Rust > 0)
Lanciare l'app da terminale per vedere stderr/log Rust:
```bash
"$BIN"
# poi premere il bottone nell'app e leggere le righe log::info/log::error ("run_diarization command triggered...", "Diarization failed...")
```
Nota: se il fallimento è "command not found", l'errore è lato JS (non nei log Rust) → affidarsi al test `strings` sopra.

## Fix atteso (caso più probabile: feature non spedita)
Buildare una build pulita di `enhance/diarization` sul Mac e reinstallarla:
```bash
cd <repo>/frontend
git checkout enhance/diarization && git pull
./clean_build.sh        # oppure ./clean_run.sh per test rapido senza installare
```
Poi ri-testare il bottone su un meeting **con file `audio.mp4`** già presente in `~/Documents/meetily-recordings/<nome>_<data>/`.
- **Attenzione modelli:** al primo run scarica ~32MB di modelli ONNX da HuggingFace in `~/Library/Application Support/<bundle>/models/diarization/` (`segmentation.onnx`, `embedding.onnx`). Serve rete. Eventi di progresso: `diarization-model-download-progress`, `diarization-progress`, `diarization-complete`.
- **Validazione qualità (caveat noto):** sullo smoke test lo split speaker era "troppo netto" (prima metà A / seconda metà B). Validare su audio reale a più voci alternate. Se over/under-clustering, il tuning è in `commands.rs` (`clustering_threshold: 0.8`) e nel core `diarization/clustering.rs` (average-linkage UPGMA, cosine).

## Vincoli / gotcha (NON violare)
- Chiave privata minisign fuori dal repo: `~/.tauri/meetily-fork.key` (password vuota) — **mai committare**. Segreto GH `TAURI_SIGNING_PRIVATE_KEY` già impostato dall'utente via web UI.
- Distribuzione solo personale, **Apple Silicon**, **niente certificato Apple a pagamento** (no notarizzazione/firma).
- Non reintrodurre il backend FastAPI archiviato. Non committare artefatti di build/test: sidecar `llama-helper`, modelli scaricati, `frontend/src-tauri/diarization_check_tmp/`, `frontend/src-tauri/src/bin/diarization_test.rs` (smoke-test con path `/tmp` Linux hardcoded, inutile sul Mac). Questi sono in `.gitignore`.
- Non disabilitare la verifica TLS.
- Concludere i blocchi di lavoro con commit + push (su `enhance/diarization`), solo se richiesto.

## Decisione aperta per l'utente (dopo il test)
Se il test conferma "feature non in `main`": decidere se **mergiare `enhance/diarization` → `main`** e rifare la Release da lì (così le release ufficiali avranno la feature), oppure continuare a testare solo con build locali finché la qualità diarization non è validata. **Chiedere all'utente prima di mergiare/rilasciare** (operazione outward-facing).

## Suggested skills
- **`tdd`** o **`systematic-debugging`** — l'utente ha chiesto esplicitamente approccio /tdd: partire dal "red test" (il comando `strings`/log) prima di toccare codice.
- **`verification-before-completion`** — verificare col vero flusso end-to-end nell'app prima di dichiarare risolto.
- **`cdx`** (o **`agy`**) — delegare l'operativo (build, edit, triage) mantenendo il ruolo PM.
- **`run`** / **`verify`** — per lanciare l'app e osservare il comportamento reale del bottone.

## File chiave
- `frontend/src/components/MeetingDetails/TranscriptButtonGroup.tsx` (bottone + invoke + toast, righe ~152-169)
- `frontend/src-tauri/src/diarization/commands.rs` (comando `run_diarization`, download modelli, allineamento, persistenza)
- `frontend/src-tauri/src/lib.rs:753-754` (registrazione `run_diarization`, `rename_speaker`)
- `frontend/src-tauri/src/diarization/` (core: `segmentation.rs`, `embedding.rs`, `clustering.rs`, `mod.rs`, `types.rs`)
