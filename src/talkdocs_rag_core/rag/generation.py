"""C2+C3 — generazione grounded: client Mistral, prompt cache-friendly, output
strutturato con citazioni, verifica citazioni e **rifiuto deterministico**.

Principi (CLAUDE.md / spec §5):
- la sicurezza/fedeltà vive nella *pipeline*, non nel testo del prompt: il rifiuto è un
  ramo di codice su ``support_threshold``, non un'istruzione al modello;
- prompt cache-friendly: prefisso [system + schema] byte-identico in testa, passaggi e
  domanda in coda; ``prompt_cache_key`` stabile per corpus/versione;
- ``usage`` sempre catturato (audit C5), inclusi i ``cached_tokens`` (M1).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from openai import BadRequestError, OpenAI, UnprocessableEntityError

from config import RagConfig
from vendor.talkdocs.services.hybrid_search import HybridSearchResult

from .guard import TermStats, abstention_signal
from .outcomes import StructuredOutcome, VerbatimOutcome
from .schema import STRUCTURED_RESPONSE_SCHEMA, StructuredAnswer
from .verbatim import verifica

logger = logging.getLogger(__name__)

# Prefisso di sistema STABILE (byte-identico ad ogni richiesta) → prompt caching provider.
SYSTEM_PREFIX = (
    "Sei un assistente che risponde ESCLUSIVAMENTE sulla base dei passaggi numerati forniti.\n"
    "Regole:\n"
    "1. Rispondi in italiano, in modo preciso e conciso.\n"
    "2. Usa solo le informazioni contenute nei passaggi numerati: non aggiungere nulla che non "
    "sia nel contesto.\n"
    "3. Cita SEMPRE la fonte mettendo il numero del passaggio tra parentesi quadre subito dopo "
    "l'affermazione che lo sostiene, es. [1] o [2][3]. Cita solo i passaggi effettivamente usati.\n"
    "4. Se i passaggi non contengono l'informazione, dichiaralo e non inventare.\n"
    "5. Restituisci un oggetto JSON con: 'answer' (testo con citazioni [n]) e 'claims' "
    "(lista di {statement, passages}), dove 'passages' elenca i numeri dei passaggi che "
    "sostengono ciascuna affermazione.\n"
    "6. Per ogni claim riporta in 'verbatim' le parole ESATTE del passaggio che sostengono "
    "l'affermazione, copiate alla lettera e non parafrasate. Se non trovi parole che la "
    "sostengano, non fare quell'affermazione."
)

_MARKER_RE = re.compile(r"\[(\d+)\]")
# Recupero della prosa dal campo "answer" di un JSON troncato: si prende tutto fino
# all'ultima virgoletta non scappata, senza pretendere che l'oggetto sia chiuso.
_ANSWER_RE = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)', re.DOTALL)


def _salvage_answer(raw: str) -> str:
    """Prosa del campo ``answer`` da un JSON non valido (di norma troncato).

    Se non si riconosce nemmeno il campo, si restituisce il grezzo: peggio di una prosa
    parziale, ma meglio di una stringa vuota che nasconderebbe il guasto.
    """
    m = _ANSWER_RE.search(raw)
    if not m:
        return raw.strip()
    frammento = m.group(1)
    # Il troncamento può cadere a metà di una sequenza di escape: si scarta la coda spuria.
    if frammento.endswith("\\") and not frammento.endswith("\\\\"):
        frammento = frammento[:-1]
    # Si richiude la sola stringa e la si fa sciogliere al decoder JSON: gestisce \uXXXX
    # (accenti sfuggiti) che un replace a mano lascerebbe come «è» sotto gli occhi
    # dell'utente.
    try:
        return json.loads(f'"{frammento}"').strip()
    except json.JSONDecodeError:
        for a, b in (('\\"', '"'), ("\\n", "\n"), ("\\t", "\t"), ("\\\\", "\\")):
            frammento = frammento.replace(a, b)
        return frammento.strip()


@dataclass
class Passage:
    n: int  # numero 1-based mostrato al modello
    chunk_id: str
    source: str
    content: str
    # Il `chunk_id` è risolvibile nell'indice? Vero per tutto ciò che viene dal retrieval;
    # falso per i passaggi del ramo meta (sezioni della scheda, blocco delle statistiche),
    # che esistono solo in memoria al momento della risposta.
    #
    # Non è un dettaglio di implementazione: una citazione a un passaggio che nessuno può
    # più rileggere **non è una citazione**. Chi giudica la vedrebbe come un rimando vuoto,
    # e l'audit non permetterebbe di difendere la risposta. Quando è falso, il testo del
    # passaggio viaggia dentro la tupla di audit (`passages_inline`).
    in_index: bool = True


@dataclass
class RagResult:
    query: str
    answer_text: str
    refused: bool
    refusal_reason: str | None
    support_score: float
    cited_passages: list[int]
    cited_chunk_ids: list[str]
    invalid_citations: list[int]  # passaggi citati fuori range → allucinazioni bloccate
    claims: list[dict]
    passages: list[Passage]
    usage: dict
    raw_output: str
    model: str
    params: dict
    # Terzo esito (C3b): né rifiuto né risposta. I passaggi sono in tema ma non contengono la
    # cosa specifica chiesta: si dichiara l'incertezza e si chiede di precisare.
    uncertain: bool = False
    missing_terms: list[str] = field(default_factory=list)
    abstention_signal: float = 0.0
    # `finish_reason == "length"`: la risposta è **incompleta**, tagliata dal tetto
    # `max_output_tokens`. Non era catturato, e una risposta troncata veniva presentata come
    # completa e contata dalle metriche come valida.
    truncated: bool = False
    finish_reason: str | None = None
    from_cache: bool = False
    cache_kind: str | None = None  # "semantic" | "provider" | None
    latency_s: float | None = None  # monotonico (perf_counter): NON avanza durante un suspend
    latency_wall_s: float | None = None  # wall-clock: se divergono, la macchina ha dormito
    extra: dict = field(default_factory=dict)
    # --- Router e guardia verbatim (incremento 1) ---
    # "pointwise" | "structured" | "uncovered" | "meta". Registrato sempre: distingue in
    # audit un POINTWISE legittimo da un fall-through del router lessicale (spec §7).
    route: str = "pointwise"
    router_signals: dict = field(default_factory=dict)
    # --- Router agentico (incremento 1b) ---
    # Chi ha deciso la route: "lexical" | "llm". La traccia della classificazione LLM
    # (proposta, usage, errore) viaggia SEPARATA da `usage`: il ramo strutturato dichiara
    # «usage vuoto per costruzione» e deve restare vero — il costo del routing è un costo,
    # ma è un altro costo, e nel report ha colonne sue.
    router_source: str = "lexical"
    router_llm: dict | None = None
    structured: StructuredOutcome | None = None
    verbatim: VerbatimOutcome | None = None
    # Motivo dell'astensione: "termini_mancanti" (guardiano IDF) | "verbatim".
    uncertain_reason: str | None = None


def _support_score(results: list[HybridSearchResult]) -> float:
    """Segnale di supporto pre-generazione: miglior similarità densa fra i top-k."""
    dense = [r.vector_score for r in results if r.vector_score is not None]
    return max(dense) if dense else 0.0


def _build_passages(results: list[HybridSearchResult], top_k: int) -> list[Passage]:
    passages = []
    for i, r in enumerate(results[:top_k], start=1):
        passages.append(Passage(n=i, chunk_id=r.chunk_id, source=r.source, content=r.content))
    return passages


def usage_dict(response) -> dict:
    """``usage`` normalizzato dalla risposta del provider (``{}`` se il provider non lo espone).

    Estratto in una funzione perché serve in **tre** punti: la risposta servita,
    l'astensione da guardia verbatim (che esce dopo aver già speso i token) e la
    chiamata di classificazione del router agentico (``rag/agentic_router.py``) — una
    sola nozione di «usage», non tre normalizzazioni che possono divergere.
    """
    u = getattr(response, "usage", None)
    if u is None:
        return {}
    details = getattr(u, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) if details else 0
    return {
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.total_tokens,
        "cached_tokens": cached or 0,
    }


def _format_passages(passages: list[Passage]) -> str:
    parts = []
    for p in passages:
        parts.append(f"[{p.n}] (documento: {p.source})\n{p.content}")
    return "\n\n---\n\n".join(parts)


class MistralGenerator:
    """Generatore grounded su La Plateforme (client OpenAI-compatible)."""

    def __init__(self, cfg: RagConfig, client: OpenAI, term_stats: TermStats | None = None):
        self.cfg = cfg
        self.client = client
        self.term_stats = term_stats

    def _uncertain_result(
        self,
        query: str,
        support: float,
        passages: list[Passage],
        segnale: float,
        mancanti: list[str],
        motivo: str = "termini_mancanti",
        verbatim: VerbatimOutcome | None = None,
        usage: dict | None = None,
        raw_output: str = "",
    ) -> RagResult:
        """Risposta di incertezza. Il **testo** non è mai generato dal modello.

        Costruirlo col modello aggiungerebbe una superficie di allucinazione proprio nel ramo
        che serve a evitarle (e un costo).

        `usage` e `raw_output` distinguono i due motivi, e la distinzione è di merito, non di
        forma. L'astensione per `termini_mancanti` esce **prima** della chiamata: lì `{}` e `""`
        sono la verità, e riempirli fingerebbe un costo mai sostenuto. L'astensione per
        `verbatim` arriva **dopo**: i token sono stati spesi comunque, e azzerarli farebbe
        sparire dalle metriche il costo delle domande peggiori — proprio quelle su cui si
        decide se la guardia conviene.

        I termini mancanti **non** compaiono nel messaggio, pur restando nell'audit. Misurato:
        il segnale non distingue *registro* da *specificità* — «premi» (df 5) e «nato» (df 7)
        sono più rari di «premiali» (df 49) — quindi su una domanda colloquiale l'astensione è
        giusta nell'esito ma la spiegazione sarebbe fuorviante («non compare "soldi"» non dice
        nulla a un cittadino). Si dichiara l'incertezza e si mostra cosa si è trovato.

        Due motivi, due testi: l'astensione da guardia verbatim arriva *dopo* la generazione e
        non è un problema di riformulazione della domanda — suggerirla sarebbe fuorviante.
        """
        if motivo == "verbatim":
            testo = (
                "Non posso rispondere con certezza: le affermazioni che avrei prodotto non "
                "risultano sostenute alla lettera dai documenti recuperati. Preferisco non "
                "presentarle piuttosto che presentarle senza una citazione verificata."
            )
        else:
            fonti = []
            for p in passages:
                if p.source not in fonti:
                    fonti.append(p.source)
            elenco = "; ".join(fonti[:5]) or "nessun documento"
            testo = (
                "Non posso rispondere con certezza: nei documenti che ho trovato non c'è un "
                "passaggio che risponda con precisione a questa domanda. "
                f"Ho trovato però documenti in tema: {elenco}. "
                "Prova a riformulare in modo più specifico — per esempio indicando l'anno, il "
                "numero della delibera, o la denominazione esatta del fondo o dell'opera."
            )
        return RagResult(
            query=query,
            answer_text=testo,
            refused=False,
            refusal_reason=None,
            support_score=support,
            cited_passages=[],
            cited_chunk_ids=[],
            invalid_citations=[],
            claims=[],
            passages=passages,
            usage=usage or {},
            raw_output=raw_output,
            model=self.cfg.mistral_model,
            params=self._params(),
            uncertain=True,
            missing_terms=mancanti,
            abstention_signal=segnale,
            uncertain_reason=motivo,
            verbatim=verbatim,
        )

    def _refusal_result(self, query: str, support: float, passages: list[Passage]) -> RagResult:
        return RagResult(
            query=query,
            answer_text=(
                "Non posso rispondere: i documenti non contengono informazioni sufficienti "
                "a sostenere una risposta."
            ),
            refused=True,
            refusal_reason=f"support_score {support:.3f} < soglia {self.cfg.support_threshold:.3f}",
            support_score=support,
            cited_passages=[],
            cited_chunk_ids=[],
            invalid_citations=[],
            claims=[],
            passages=passages,
            usage={},
            raw_output="",
            model=self.cfg.mistral_model,
            params=self._params(),
        )

    def _params(self) -> dict:
        return {
            "model": self.cfg.mistral_model,
            "temperature": self.cfg.llm_temperature,
            "max_tokens": self.cfg.max_output_tokens,
            "seed": self.cfg.llm_seed,
            "top_k": self.cfg.rag_top_k,
        }

    def _messages(self, passages: list[Passage], query: str) -> list[dict]:
        # Prefisso stabile in testa (system); variabile (passaggi + domanda) in coda.
        user = (
            f"Passaggi disponibili:\n\n{_format_passages(passages)}\n\n---\n\n"
            f"Domanda: {query}\n\n"
            "Rispondi in JSON secondo lo schema, citando i passaggi con [n]."
        )
        return [
            {"role": "system", "content": SYSTEM_PREFIX},
            {"role": "user", "content": user},
        ]

    def _call(self, messages: list[dict], cache_key: str):
        """Chiamata con output strutturato; fallback json_object se lo schema è rifiutato.

        Il fallback intercetta **solo** il rifiuto dello schema (400/422). Un ``except
        Exception`` largo qui è una trappola: trasforma un timeout o un errore di rete in
        una *seconda* richiesta completa con la sua scala di retry, raddoppiando l'attesa
        e mascherando il guasto da "schema non supportato". Timeout, 429 e 5xx devono
        propagare: li gestisce già il retry con backoff dell'SDK.
        """
        common = dict(
            model=self.cfg.mistral_model,
            messages=messages,
            temperature=self.cfg.llm_temperature,
            max_tokens=self.cfg.max_output_tokens,
        )
        if self.cfg.llm_seed is not None:
            common["random_seed"] = self.cfg.llm_seed
        # prompt_cache_key: raggruppa richieste con lo stesso prefisso (stesso corpus).
        extra_body = {"prompt_cache_key": cache_key}
        try:
            return self.client.chat.completions.create(
                response_format=STRUCTURED_RESPONSE_SCHEMA, extra_body=extra_body, **common
            )
        except (BadRequestError, UnprocessableEntityError) as exc:
            # Fallback: JSON object mode (lo schema è comunque descritto nel system prefix).
            logger.warning(
                "Schema strutturato rifiutato dal provider (%s) — fallback json_object",
                exc.__class__.__name__,
            )
            return self.client.chat.completions.create(
                response_format={"type": "json_object"}, extra_body=extra_body, **common
            )

    def _parse(self, raw: str, n_passages: int) -> tuple[StructuredAnswer, list[int], list[int]]:
        """Parsing + verifica citazioni. Ritorna (structured, validi, invalidi)."""
        try:
            data = json.loads(raw)
            structured = StructuredAnswer.model_validate(data)
        except Exception:
            # JSON non valido — tipicamente **troncato** dal tetto `max_output_tokens`: le
            # domande vaghe producono risposte lunghe ed enumerative che sbattono contro il
            # limite a metà dell'array `claims`.
            #
            # Il degrado a `answer=raw` era un difetto: riversava JSON grezzo con graffe e
            # virgolette nel campo mostrato all'utente, e il regex delle citazioni trovava
            # comunque i marcatori dentro quel testo — così le metriche automatiche
            # registravano «5 citazioni valide, 0 invalide» su un output inutilizzabile
            # (misurato su va-05, run eval-20260805T111224Z).
            #
            # Si recupera la prosa del campo `answer` anche da JSON incompleto.
            structured = StructuredAnswer(answer=_salvage_answer(raw), claims=[])

        # Citazioni dal testo answer + dai claims.
        cited = []
        for m in _MARKER_RE.findall(structured.answer):
            cited.append(int(m))
        for c in structured.claims:
            cited.extend(c.passages)

        valid = sorted({n for n in cited if 1 <= n <= n_passages})
        invalid = sorted({n for n in cited if not (1 <= n <= n_passages)})
        return structured, valid, invalid

    def generate(self, query: str, results: list[HybridSearchResult], cache_key: str) -> RagResult:
        passages = _build_passages(results, self.cfg.rag_top_k)
        support = _support_score(results)

        # --- Rifiuto DETERMINISTICO (ramo di codice, non prompt) ---
        if support < self.cfg.support_threshold or not passages:
            return self._refusal_result(query, support, passages)

        # --- Astensione (C3b): in tema ma senza la cosa chiesta ---
        # Ortogonale al support_score, che misura vicinanza di argomento e qui non aiuta.
        segnale, mancanti = abstention_signal(query, [p.content for p in passages], self.term_stats)
        if self.cfg.abstention_idf_threshold > 0 and segnale >= self.cfg.abstention_idf_threshold:
            return self._uncertain_result(query, support, passages, segnale, mancanti)

        return self.genera_da_passaggi(
            query, passages, cache_key, support=support, segnale=segnale, mancanti=mancanti
        )

    def genera_da_passaggi(
        self,
        query: str,
        passages: list[Passage],
        cache_key: str,
        support: float = 1.0,
        segnale: float = 0.0,
        mancanti: list[str] | None = None,
    ) -> RagResult:
        """Generazione grounded su passaggi **già costruiti**, non venuti dal retrieval.

        La usa il ramo meta (`structured.service.serve_meta`), dove i «passaggi» sono le
        sezioni della scheda del corpus e il blocco di statistiche calcolate. Il contratto
        è identico — stesso `SYSTEM_PREFIX` (quindi stesso prefisso cache-friendly), stesso
        schema di output, stessa verifica delle citazioni e stessa guardia verbatim — e
        questo è il punto: **una cifra che il modello scrive senza che stia nella scheda o
        nel blocco calcolato risulta non verificata**, esattamente come su una delibera.

        Le due guardie che dipendono dal retrieval (soglia di supporto e guardiano IDF)
        non si applicano: qui i passaggi non sono stati recuperati, sono dati.
        """
        mancanti = mancanti or []
        messages = self._messages(passages, query)
        response = self._call(messages, cache_key)
        scelta = response.choices[0]
        raw = scelta.message.content or ""
        finish_reason = getattr(scelta, "finish_reason", None)
        truncated = finish_reason == "length"
        if truncated:
            logger.warning(
                "Risposta TRONCATA dal tetto max_tokens=%s: output incompleto per «%.60s»",
                self.cfg.max_output_tokens,
                query,
            )

        structured, valid, invalid = self._parse(raw, len(passages))
        cited_chunk_ids = [passages[n - 1].chunk_id for n in valid]
        # L'usage si estrae **qui**, non al momento di costruire la risposta servita: la
        # guardia verbatim può uscire prima, e i token della chiamata sono già stati spesi.
        usage = usage_dict(response)

        # --- Guardia verbatim (C3c): le parole dichiarate esistono dove il modello dice? ---
        # A differenza dell'astensione IDF, questa arriva **dopo** la chiamata: l'astensione
        # porta con sé `usage` e `raw_output` reali, altrimenti con la guardia accesa il costo
        # sparirebbe dalle metriche proprio sulle domande peggiori.
        esito_verbatim = None
        if self.cfg.verbatim_enabled:
            esito_verbatim = verifica(
                [c.model_dump() for c in structured.claims], passages, self.cfg.verbatim_min_chars
            )
            if (
                self.cfg.verbatim_min_valid_ratio > 0
                and esito_verbatim.valid_ratio is not None
                and esito_verbatim.valid_ratio < self.cfg.verbatim_min_valid_ratio
            ):
                logger.warning(
                    "Verbatim sotto soglia (%.2f < %.2f) su «%.60s»: astensione",
                    esito_verbatim.valid_ratio,
                    self.cfg.verbatim_min_valid_ratio,
                    query,
                )
                return self._uncertain_result(
                    query, support, passages, segnale, mancanti,
                    motivo="verbatim", verbatim=esito_verbatim,
                    usage=usage, raw_output=raw,
                )

        return RagResult(
            query=query,
            answer_text=structured.answer,
            refused=False,
            refusal_reason=None,
            support_score=support,
            cited_passages=valid,
            cited_chunk_ids=cited_chunk_ids,
            invalid_citations=invalid,
            claims=[c.model_dump() for c in structured.claims],
            passages=passages,
            usage=usage,
            raw_output=raw,
            model=self.cfg.mistral_model,
            params=self._params(),
            # Registrati anche quando NON hanno fatto scattare l'astensione: servono a ritarare
            # la soglia su run future senza rigiocare tutto.
            missing_terms=mancanti,
            abstention_signal=segnale,
            truncated=truncated,
            finish_reason=finish_reason,
            cache_kind="provider" if usage.get("cached_tokens") else None,
            verbatim=esito_verbatim,
        )
