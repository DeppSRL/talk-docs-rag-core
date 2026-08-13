# talkdocs-rag-core

Nucleo RAG *grounded* estratto dal banco di misura `parlaconme-mistral-poc`: retrieval
ibrido (vettoriale + keyword italiano, fusione RRF), **rifiuto deterministico** su soglia di
supporto, **astensione** per segnale IDF, **verifica delle citazioni a valle** della
generazione, guardia verbatim, tupla di audit rigiocabile, e l'**harness di valutazione**
che misura tutto questo.

Non contiene strato applicativo: né FastAPI, né auth, né multi-index, né UI. Quelli sono del
consumatore — è il motivo per cui la demo si costruisce in talk-docs, dove esistono già.

## Perché esiste

Non per riusare codice: per **quotare in fretta con costi certi**. Il criterio di successo è
un secondo corpus in piedi e misurato **in una giornata**, non un'interfaccia elegante.

## Installazione

Dipendenza git pinnata (mai un branch: una libreria che decide le soglie di rifiuto non può
cambiare sotto i piedi di una misura):

```toml
dependencies = [
  "talkdocs-rag-core @ git+https://github.com/DeppSRL/parlaconme-mistral-poc@<sha>#subdirectory=packages/talkdocs-rag-core",
]
```

## Uso

```python
from talkdocs_rag_core import RagConfig, build_pipeline, run_ingest

cfg = RagConfig.from_env()                    # oppure RagConfig(...) esplicita, per indice
report = await run_ingest(cfg)                 # corpus → Chroma + Whoosh + manifest
pipeline = await build_pipeline(cfg)
res = await pipeline.ask("...", use_cache=True)

res.refused, res.refusal_reason               # il rifiuto è un esito, non un errore
res.answer_text, res.cited_passages           # citazioni già verificate dalla pipeline
res.support_score, res.usage                  # per l'audit e per il costo
```

## Le due cose da non ereditare

1. **La soglia di supporto.** `support_threshold` è un coseno su un modello di embedding
   specifico: 0,79 vale su `mistral-embed-2312` a 1024 dimensioni **sul corpus delle
   delibere**. Su un corpus nuovo si ri-tara con `talkdocs_rag_core.eval.tara_soglia`. È la
   procedura, non il numero, il pezzo che rende una giornata sufficiente.
2. **Il modello di embedding, a caldo.** Cambiarlo su un indice esistente invalida la
   collection (1024 e 1536 dimensioni non sono compatibili): è una scelta **alla creazione**
   dell'indice, e cambiarla è un re-ingest esplicito.

## Provenienza

`src/talkdocs_rag_core/retrieval/` viene dal nucleo di talk-docs, commit
`6dd976c946bc5ef296dd4f6c8e7b00a242dc6c2b` — vedi il `NOTICE` lì accanto. Da questo
pacchetto in avanti la direzione si inverte: era vendorizzato *da* talk-docs, ora è il codice
che talk-docs **consuma**.
