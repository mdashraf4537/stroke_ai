"""Clinical RAG & Google/Web Search Fallback Client for Stroke Flask App.

Connects to the FastAPI Hybrid RAG microservice (http://127.0.0.1:8000/api/v1)
and provides:
1. Interactive clinical Q&A grounded in ingested AHA/ASA Stroke Guidelines.
2. Automatic Google/Web + NCBI PubMed RAG fallback when a doctor's question is
   not covered by the local knowledge base.
3. Direct local index + web fallback resilience if the port 8000 daemon is offline.
"""

import html
import os
import re
import sqlite3
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

RAG_API_URL = os.environ.get(
    "RAG_API_URL", "https://research-assistant-pa3v.onrender.com/api/v1"
)
RAG_USER_EMAIL = os.environ.get(
    "RAG_USER_EMAIL", "mdashraf228144@gmail.com"
)
RAG_PROJECT_ROOT = Path(
    os.environ.get(
        "RAG_PROJECT_ROOT",
        r"C:\Users\ASUS\Documents\Codex\2026-08-10\research_assistant",
    )
)
RAG_ENV_PATH = RAG_PROJECT_ROOT / ".env"
RAG_LEXICAL_DB = RAG_PROJECT_ROOT / "data" / "lexical.db"

KB_SYSTEM_PROMPT = """You are a rigorous clinical stroke and research assistant.
Answer only from the supplied evidence. Cite every factual claim using the supplied
bracketed citation IDs, such as [S1], [S2]. Do not invent citations or use knowledge not
present in the evidence. If the supplied evidence does not contain the answer to the
doctor's specific question, state clearly that the supplied evidence is insufficient."""

WEB_RAG_SYSTEM_PROMPT = """You are an expert clinical stroke decision-support assistant.
The clinician's question was not covered in the local PDF knowledge base, so live Google/Web and PubMed medical literature evidence was retrieved.
Answer the clinician's question clearly, accurately, and concisely using the supplied web and medical literature evidence below.
If patient case context is provided, tailor the clinical interpretation to that patient while citing the retrieved guidelines/literature.
Cite every factual claim using bracketed citation IDs such as [S1], [S2].
Preserve exact clinical thresholds (e.g., blood pressure targets in mmHg, time windows in hours, dosages). Do not invent citations."""


def _load_rag_env() -> dict[str, str]:
    env_vars: dict[str, str] = {}
    if RAG_ENV_PATH.exists():
        for line in RAG_ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env_vars[k.strip()] = v.strip().strip('"').strip("'")
    return env_vars


def _is_insufficient_answer(answer: str, citations: list) -> bool:
    if not citations or not answer:
        return True
    if not re.search(r"\[S\d+\]", answer):
        return True
    lowered = answer.lower()
    insufficiency_phrases = (
        "could not find relevant evidence",
        "insufficient",
        "not present in the",
        "not mentioned in the",
        "does not contain",
        "does not provide",
        "do not contain",
        "do not provide",
        "no information",
        "cannot answer",
        "not covered",
        "supplied evidence does not",
        "provided evidence does not",
        "evidence does not specify",
        "unavailable",
        "returned an error",
    )
    return any(phrase in lowered for phrase in insufficiency_phrases)


def _strip_html(raw: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", raw or "")
    cleaned = html.unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _extract_url_from_excerpt(excerpt: str) -> str:
    match = re.search(r"(?:Source URL|URL):\s*(https?://\S+)", excerpt or "")
    return match.group(1).strip() if match else ""


def _enrich_citations(citations: list[dict]) -> list[dict]:
    enriched = []
    for idx, c in enumerate(citations, start=1):
        cid = c.get("citation_id") or c.get("id") or f"S{idx}"
        excerpt = c.get("excerpt", "")
        url = c.get("url") or _extract_url_from_excerpt(excerpt)
        enriched.append(
            {
                "citation_id": cid,
                "filename": c.get("filename", "Clinical Source"),
                "page_number": c.get("page_number", 1),
                "excerpt": excerpt,
                "url": url,
            }
        )
    return enriched


def _search_local_bm25(query: str, limit: int = 8) -> list[dict]:
    """Query the local SQLite FTS5 BM25 index directly if FastAPI server is offline."""
    if not RAG_LEXICAL_DB.exists():
        return []
    tokens = [
        re.sub(r"[^a-zA-Z0-9]", "", tok).lower()
        for tok in query.split()
        if len(re.sub(r"[^a-zA-Z0-9]", "", tok)) >= 3
    ]
    stop = {
        "the", "and", "for", "with", "what", "can", "this", "patient", "from",
        "are", "was", "were", "that", "have", "has", "had", "but", "not", "according",
        "doctor", "followup", "question", "context", "case", "ago", "hours",
    }
    keywords = [t for t in tokens if t not in stop][:12]
    if not keywords:
        return []
    fts_query = " OR ".join(f'"{kw}"' for kw in keywords)
    try:
        conn = sqlite3.connect(str(RAG_LEXICAL_DB))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT chunk_id, text, filename, page_number, bm25(lexical_chunks) AS rank
            FROM lexical_chunks
            WHERE lexical_chunks MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
        conn.close()
        return [
            {
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "filename": r["filename"],
                "page_number": int(r["page_number"]),
            }
            for r in rows
        ]
    except Exception:
        return []


def _call_deepseek(system_prompt: str, user_prompt: str) -> str:
    env = _load_rag_env()
    api_key = env.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
    model = env.get("DEEPSEEK_MODEL", "deepseek-chat")
    if not api_key:
        return ""
    for model_name in [model, "deepseek-chat"]:
        try:
            resp = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model_name,
                    "temperature": 0.1,
                    "max_tokens": 1200,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=35.0,
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"].get("content", "")
                if content:
                    return content.strip()
        except Exception:
            continue
    return ""


def _search_web_and_pubmed(clean_query: str) -> list[dict]:
    """Retrieve live web & medical literature snippets from PubMed, DuckDuckGo, Google RSS, and Wikipedia."""
    snippets: list[dict] = []
    seen_urls: set[str] = set()

    def add_snippet(title: str, url: str, text: str, label: str):
        norm = url.strip().lower()
        if text and norm not in seen_urls:
            seen_urls.add(norm)
            snippets.append(
                {
                    "title": title.strip(),
                    "url": url.strip(),
                    "snippet": text.strip(),
                    "source_label": label,
                }
            )

    # 1. NCBI PubMed E-Utilities (Authoritative medical guidelines & stroke trials)
    try:
        esearch = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": clean_query,
                "retmode": "json",
                "retmax": 3,
                "sort": "relevance",
            },
            timeout=10.0,
        )
        if esearch.status_code == 200:
            ids = esearch.json().get("esearchresult", {}).get("idlist", [])
            if ids:
                efetch = requests.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                    params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
                    timeout=10.0,
                )
                if efetch.status_code == 200:
                    root = ET.fromstring(efetch.text)
                    for art in root.findall(".//PubmedArticle"):
                        pmid = art.findtext(".//PMID") or ""
                        title = art.findtext(".//ArticleTitle") or "PubMed Study"
                        abs_parts = [
                            (el.attrib.get("Label", "") + ": " if el.attrib.get("Label") else "")
                            + "".join(el.itertext()).strip()
                            for el in art.findall(".//AbstractText")
                        ]
                        abstract = " ".join(p for p in abs_parts if p).strip()
                        if abstract:
                            add_snippet(
                                title,
                                f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                                abstract[:850],
                                "PubMed / NIH",
                            )
    except Exception:
        pass

    # 2. DuckDuckGo Web Search
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        }
        ddg = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": clean_query},
            headers=headers,
            timeout=10.0,
        )
        if ddg.status_code == 200:
            blocks = re.findall(
                r'<div class="result__body">(.*?)</div>\s*</div>',
                ddg.text,
                flags=re.DOTALL,
            )
            for block in blocks[:4]:
                t_match = re.search(
                    r'class="result__a"[^>]* href="([^"]+)"[^>]*>(.*?)</a>',
                    block,
                    flags=re.DOTALL,
                )
                s_match = re.search(
                    r'class="result__snippet"[^>]*>(.*?)</a>|class="result__snippet"[^>]*>(.*?)</div>',
                    block,
                    flags=re.DOTALL,
                )
                if not t_match:
                    continue
                raw_url = html.unescape(t_match.group(1))
                if "uddg=" in raw_url:
                    qs = urllib.parse.parse_qs(urllib.parse.urlparse(raw_url).query)
                    raw_url = qs.get("uddg", [raw_url])[0]
                title = _strip_html(t_match.group(2))
                raw_s = (s_match.group(1) or s_match.group(2) or "") if s_match else ""
                snip = _strip_html(raw_s)
                domain = urllib.parse.urlparse(raw_url).netloc or "Web"
                if title and snip:
                    add_snippet(title, raw_url, snip, f"Google/Web ({domain})")
    except Exception:
        pass

    # 3. Google Search RSS
    try:
        grss = requests.get(
            "https://news.google.com/rss/search",
            params={"q": clean_query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
            timeout=8.0,
        )
        if grss.status_code == 200:
            root = ET.fromstring(grss.text)
            for item in root.findall(".//item")[:3]:
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                desc = _strip_html(item.findtext("description") or "")
                source = (item.findtext("source") or "Google Search").strip()
                if title:
                    add_snippet(
                        title,
                        link,
                        desc if len(desc) > 25 else title,
                        f"Google ({source})",
                    )
    except Exception:
        pass

    # 4. Wikipedia Medical Encyclopedia
    try:
        wiki = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": clean_query,
                "srlimit": 2,
                "format": "json",
            },
            headers={"User-Agent": "StrokeClinicalRAG/1.0"},
            timeout=8.0,
        )
        if wiki.status_code == 200:
            for item in wiki.json().get("query", {}).get("search", []):
                title = item.get("title", "")
                snip = _strip_html(item.get("snippet", ""))
                if title and snip:
                    url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                    add_snippet(title, url, snip, "Medical Encyclopedia")
    except Exception:
        pass

    return snippets[:6]


def _run_web_rag_fallback(raw_question: str, full_prompt: str) -> dict:
    """Execute Google/Web + PubMed search and synthesize a grounded RAG response."""
    snippets = _search_web_and_pubmed(raw_question)
    if not snippets:
        return {
            "answer": "Could not retrieve relevant clinical guidelines from the local knowledge base or live web search.",
            "citations": [],
            "source_mode": "google_search_rag",
            "retrieval_candidate_count": 0,
        }

    citations = []
    evidence_blocks = []
    for idx, s in enumerate(snippets, start=1):
        cid = f"S{idx}"
        fname = f"{s['source_label']}: {s['title'][:85]}"
        excerpt = f"{s['snippet']}\nSource URL: {s['url']}"
        citations.append(
            {
                "citation_id": cid,
                "filename": fname,
                "page_number": 1,
                "excerpt": excerpt,
                "url": s["url"],
            }
        )
        evidence_blocks.append(f"[{cid}] {fname}\n{excerpt}")

    evidence_str = "\n\n".join(evidence_blocks)
    answer = _call_deepseek(
        WEB_RAG_SYSTEM_PROMPT,
        f"Question:\n{full_prompt}\n\nRetrieved Web & Medical Literature Evidence:\n{evidence_str}",
    )
    if not answer:
        # Extractive summary fallback if LLM API key is unreachable
        answer = (
            "Based on live medical literature and web search:\n\n"
            + "\n\n".join(
                f"• **{s['title']}**: {s['snippet'][:280]}... [{f'S{i}'}]"
                for i, s in enumerate(snippets[:3], start=1)
            )
        )

    return {
        "answer": answer,
        "citations": citations,
        "source_mode": "google_search_rag",
        "retrieval_candidate_count": len(citations),
    }


def ask_clinical_qa(
    question: str,
    patient_context: str = "",
    search_mode: str = "auto",
) -> dict:
    """
    Interactive Clinical QA with Knowledge Base RAG + automatic Google/Web Search RAG fallback.
    search_mode: "auto" (KB first -> Web RAG fallback) | "kb_only" | "web_only"
    """
    clean_question = (question or "").strip()
    if not clean_question:
        return {
            "answer": "Please enter a clinical question.",
            "citations": [],
            "source_mode": "knowledge_base",
            "retrieval_candidate_count": 0,
        }

    if patient_context and patient_context.strip():
        full_query = (
            f"Patient Case Context: {patient_context.strip()}\n"
            f"Doctor's Follow-up Question: {clean_question}"
        )
    else:
        full_query = clean_question

    # 1. If explicitly forced to Google/Web search mode
    if search_mode == "web_only":
        return _run_web_rag_fallback(clean_question, full_query)

    # 2. Try calling the primary FastAPI Hybrid RAG microservice on port 8000
    try:
        response = requests.post(
            f"{RAG_API_URL}/ask",
            json={
                "question": full_query,
                "search_mode": search_mode,
                "user_email": RAG_USER_EMAIL,
            },
            timeout=90.0,
        )
        if response.status_code == 200:
            data = response.json()
            ans = data.get("answer", "")
            cits = _enrich_citations(data.get("citations", []))
            s_mode = data.get("source_mode", "knowledge_base")
            cand_count = data.get("retrieval_candidate_count", len(cits))

            if s_mode == "google_search_rag" or search_mode == "kb_only":
                return {
                    "answer": ans,
                    "citations": cits,
                    "source_mode": s_mode,
                    "retrieval_candidate_count": cand_count,
                }

            if not _is_insufficient_answer(ans, cits):
                return {
                    "answer": ans,
                    "citations": cits,
                    "source_mode": "knowledge_base",
                    "retrieval_candidate_count": cand_count,
                }
            # Local KB returned insufficient evidence -> fall through to Web RAG!
            return _run_web_rag_fallback(clean_question, full_query)
    except Exception:
        pass

    # 3. Direct local SQLite BM25 knowledge base lookup if FastAPI daemon is not running
    local_chunks = _search_local_bm25(clean_question, limit=6)
    if local_chunks:
        citations = [
            {
                "citation_id": f"S{idx}",
                "filename": c["filename"],
                "page_number": c["page_number"],
                "excerpt": c["text"],
                "url": "",
            }
            for idx, c in enumerate(local_chunks, start=1)
        ]
        evidence_str = "\n\n".join(
            f"[{c['citation_id']}] {c['filename']}, page {c['page_number']}\n{c['excerpt']}"
            for c in citations
        )
        kb_answer = _call_deepseek(
            KB_SYSTEM_PROMPT,
            f"Question:\n{full_query}\n\nEvidence:\n{evidence_str}",
        )
        if kb_answer and not _is_insufficient_answer(kb_answer, citations):
            return {
                "answer": kb_answer,
                "citations": citations,
                "source_mode": "knowledge_base",
                "retrieval_candidate_count": len(citations),
            }
        if search_mode == "kb_only":
            return {
                "answer": kb_answer or "Insufficient evidence in local AHA/ASA knowledge base.",
                "citations": citations,
                "source_mode": "knowledge_base",
                "retrieval_candidate_count": len(citations),
            }

    # 4. Automatic fallback to Google / Web + PubMed RAG
    return _run_web_rag_fallback(clean_question, full_query)


def query_clinical_guidelines(question: str) -> dict:
    """Query the RAG microservice for AHA/ASA evidence-grounded guidance on prediction pages."""
    return ask_clinical_qa(question=question, patient_context="", search_mode="auto")
