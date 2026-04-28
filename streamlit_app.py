"""
A-MEM: Agentic Memory System for LLM Agents
Based on: "A-Mem: Agentic Memory for LLM Agents" (arXiv:2502.12110)

Implements the full pipeline:
  1. Note Construction  (Section 3.1)
  2. Link Generation    (Section 3.2)
  3. Memory Evolution   (Section 3.3)
  4. Memory Retrieval   (Section 3.4)

Usage:
    pip install openai sentence-transformers numpy

    from amem import AMem
    mem = AMem(openai_api_key="sk-...")
    mem.add("I love hiking in the mountains on weekends.")
    mem.add("I went on a trail run last Saturday and it was exhausting.")
    results = mem.retrieve("outdoor activities")
    for r in results:
        print(r.summary())
"""

import json
import uuid
import numpy as np
from datetime import datetime
from typing import Optional
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import streamlit as st


# ══════════════════════════════════════════════════════════════════════════════
#  MEMORY NOTE  (Equation 1 in the paper)
#  m_i = {c_i, t_i, K_i, G_i, X_i, e_i, L_i}
# ══════════════════════════════════════════════════════════════════════════════

class MemoryNote:
    def __init__(self, content: str):
        self.id: str = str(uuid.uuid4())[:8]
        self.content: str = content                        # c_i  – raw interaction
        self.timestamp: str = datetime.now().isoformat()   # t_i
        self.keywords: list[str] = []                      # K_i  – LLM-generated
        self.tags: list[str] = []                          # G_i  – LLM-generated
        self.context: str = ""                             # X_i  – LLM-generated
        self.embedding: Optional[np.ndarray] = None        # e_i  – dense vector
        self.links: list[str] = []                         # L_i  – linked note IDs

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "timestamp": self.timestamp,
            "keywords": self.keywords,
            "tags": self.tags,
            "context": self.context,
            "links": self.links,
        }

    def summary(self) -> str:
        return (
            f"[{self.id}] {self.context}\n"
            f"  keywords : {', '.join(self.keywords)}\n"
            f"  tags     : {', '.join(self.tags)}\n"
            f"  links    : {', '.join(self.links) or 'none'}\n"
            f"  content  : {self.content}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPT TEMPLATES  (Appendix B in the paper)
# ══════════════════════════════════════════════════════════════════════════════

PS1_NOTE_CONSTRUCTION = """\
Generate a structured analysis of the following content by:
1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
2. Extracting core themes and contextual elements
3. Creating relevant categorical tags

Format the response as a JSON object with NO extra text:
{
  "keywords": ["kw1", "kw2", ...],
  "context": "one sentence summarising the main topic, key points, and intended audience/purpose",
  "tags": ["tag1", "tag2", ...]
}

Rules:
- At least 3 keywords, avoid speaker names or timestamps.
- At least 3 tags covering domain, format, and type.
- Do not include markdown fences.

Content for analysis:
TIMESTAMP: {timestamp}
CONTENT: {content}
"""

PS2_LINK_GENERATION = """\
You are an AI memory agent. Decide whether the NEW memory should be linked to any of the NEIGHBOUR memories.

NEW MEMORY:
  context  : {context}
  content  : {content}
  keywords : {keywords}

NEIGHBOUR MEMORIES:
{neighbours}

Return ONLY valid JSON with no extra text:
{{
  "should_link": true | false,
  "linked_ids": ["id1", "id2", ...]   // subset of neighbour IDs worth linking; empty list if none
}}
"""

PS3_MEMORY_EVOLUTION = """\
You are an AI memory evolution agent. Based on the NEW memory, update the NEIGHBOUR memories if their
context or tags can be enriched.

NEW MEMORY:
  context  : {context}
  content  : {content}
  keywords : {keywords}

NEIGHBOUR MEMORIES (id | context | tags):
{neighbours}

Return ONLY valid JSON with no extra text. The list order must match the order of neighbour memories above:
{{
  "should_evolve": true | false,
  "updates": [
    {{
      "id": "neighbour_id",
      "new_context": "updated context sentence or empty string to keep original",
      "new_tags": ["tag1", "tag2"]   // updated tag list or empty list to keep original
    }},
    ...
  ]
}}
"""


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Equation (4): s_{n,j} = (e_n · e_j) / (|e_n| |e_j|)"""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _safe_json(text: str) -> dict:
    """Strip markdown fences and parse JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# ══════════════════════════════════════════════════════════════════════════════
#  A-MEM CORE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class AMem:
    """
    Agentic Memory for LLM Agents.

    Parameters
    ----------
    openai_api_key : str
        Your OpenAI API key.
    llm_model : str
        OpenAI chat model to use for note construction, link generation,
        and memory evolution. Default: "gpt-4o-mini".
    embed_model : str
        SentenceTransformer model for dense embeddings (Equation 3).
        Default: "all-MiniLM-L6-v2"  (same as the paper).
    top_k : int
        Number of nearest neighbours to retrieve during link generation
        and memory evolution. Default: 10 (paper default).
    verbose : bool
        Print step-by-step logs.
    """

    def __init__(
        self,
        openai_api_key: str,
        llm_model: str = "gpt-4o-mini",
        embed_model: str = "all-MiniLM-L6-v2",
        top_k: int = 10,
        verbose: bool = True,
    ):
        self.client = OpenAI(api_key=openai_api_key)
        self.llm_model = llm_model
        self.top_k = top_k
        self.verbose = verbose
        self.memory: dict[str, MemoryNote] = {}   # M = {m_1, ..., m_N}

        if verbose:
            print(f"[A-MEM] Loading embedding model '{embed_model}' …")
        self.encoder = SentenceTransformer(embed_model)
        if verbose:
            print("[A-MEM] Ready.\n")

    # ──────────────────────────────────────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def add(self, content: str) -> MemoryNote:
        """
        Full A-MEM ingestion pipeline for one new memory:
          Step 1 – Note Construction
          Step 2 – Link Generation
          Step 3 – Memory Evolution
        Returns the created MemoryNote.
        """
        # ── Step 1: Note Construction (Section 3.1) ──────────────────────────
        note = self._construct_note(content)
        self._log(f"[1] Note constructed  id={note.id}")
        self._log(f"    context  : {note.context}")
        self._log(f"    keywords : {note.keywords}")
        self._log(f"    tags     : {note.tags}")

        # Compute embedding: e_i = f_enc[concat(c, K, G, X)]  (Equation 3)
        concat_text = " ".join([note.content] + note.keywords + note.tags + [note.context])
        note.embedding = self.encoder.encode(concat_text, normalize_embeddings=True)

        # Store note (before linking so neighbours can see it)
        self.memory[note.id] = note

        if len(self.memory) > 1:
            # ── Step 2: Link Generation (Section 3.2) ────────────────────────
            neighbours = self._get_top_k_neighbours(note)
            self._log(f"[2] Link generation  neighbours={[n.id for n in neighbours]}")
            linked_ids = self._generate_links(note, neighbours)
            note.links = linked_ids
            # Update reverse links
            for nid in linked_ids:
                if nid in self.memory and note.id not in self.memory[nid].links:
                    self.memory[nid].links.append(note.id)
            self._log(f"    linked to: {linked_ids or 'none'}")

            # ── Step 3: Memory Evolution (Section 3.3) ───────────────────────
            self._log("[3] Memory evolution …")
            self._evolve_neighbours(note, neighbours)

        self._log("")
        return note

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[MemoryNote]:
        """
        Retrieve the top-k most relevant memories for a query (Section 3.4).
        Also returns memories linked to the top result (Zettelkasten 'box' effect).

        Equations (8–10):
          e_q = f_enc(q)
          s_{q,i} = cosine(e_q, e_i)
          M_retrieved = top-k ranked notes
        """
        if not self.memory:
            return []

        k = top_k or self.top_k
        query_emb = self.encoder.encode(query, normalize_embeddings=True)

        scored = [
            (cosine_similarity(query_emb, note.embedding), note)
            for note in self.memory.values()
            if note.embedding is not None
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top_notes = [note for _, note in scored[:k]]

        # Include linked notes from the top result (the "box" concept)
        if top_notes:
            best = top_notes[0]
            for lid in best.links:
                if lid in self.memory and self.memory[lid] not in top_notes:
                    top_notes.append(self.memory[lid])

        return top_notes

    def chat(self, user_message: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """
        Answer a user message using retrieved memories as context.
        This mimics the agent interaction loop in the paper.
        """
        retrieved = self.retrieve(user_message)
        memory_context = "\n\n".join(
            f"[Memory {i+1}]\n{note.summary()}" for i, note in enumerate(retrieved)
        ) if retrieved else "No relevant memories found."

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Relevant memories from your long-term memory:\n{memory_context}\n\n"
                    f"User question: {user_message}"
                ),
            },
        ]
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
        )
        return response.choices[0].message.content

    def show_all(self):
        """Return a string representation of every note in memory."""
        if not self.memory:
            return "[A-MEM] Memory is empty."
        output = f"[A-MEM] {len(self.memory)} notes in memory:\n\n"
        for note in self.memory.values():
            output += note.summary() + "\n\n"
        return output

    def dump_json(self) -> str:
        """Return all memory notes as a JSON string."""
        return json.dumps([n.to_dict() for n in self.memory.values()], indent=2)

    # ──────────────────────────────────────────────────────────────────────────
    #  PRIVATE METHODS
    # ──────────────────────────────────────────────────────────────────────────

    def _llm(self, prompt: str) -> str:
        """Single LLM call."""
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ── Step 1 ────────────────────────────────────────────────────────────────

    def _construct_note(self, content: str) -> MemoryNote:
        """
        Section 3.1 – Note Construction.
        K_i, G_i, X_i ← LLM(c_i ∥ t_i ∥ P_s1)   (Equation 2)
        """
        note = MemoryNote(content)
        prompt = PS1_NOTE_CONSTRUCTION.format(
            timestamp=note.timestamp,
            content=content,
        )
        raw = self._llm(prompt)
        try:
            parsed = _safe_json(raw)
            note.keywords = parsed.get("keywords", [])
            note.tags = parsed.get("tags", [])
            note.context = parsed.get("context", "")
        except Exception as e:
            self._log(f"    [WARN] Note construction JSON parse failed: {e}")
            note.keywords = []
            note.tags = []
            note.context = content[:120]
        return note

    # ── Step 2 ────────────────────────────────────────────────────────────────

    def _get_top_k_neighbours(self, note: MemoryNote) -> list[MemoryNote]:
        """
        Equations (4–5): retrieve top-k nearest neighbours by cosine similarity.
        Excludes the note itself.
        """
        others = [n for n in self.memory.values() if n.id != note.id and n.embedding is not None]
        if not others:
            return []
        scored = [(cosine_similarity(note.embedding, n.embedding), n) for n in others]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:self.top_k]]

    def _generate_links(self, note: MemoryNote, neighbours: list[MemoryNote]) -> list[str]:
        """
        Equation (6): L_i ← LLM(m_n ∥ M^n_near ∥ P_s2)
        """
        if not neighbours:
            return []
        neighbour_text = "\n".join(
            f"  id={n.id} | context={n.context} | keywords={n.keywords}"
            for n in neighbours
        )
        prompt = PS2_LINK_GENERATION.format(
            context=note.context,
            content=note.content,
            keywords=note.keywords,
            neighbours=neighbour_text,
        )
        raw = self._llm(prompt)
        try:
            parsed = _safe_json(raw)
            if parsed.get("should_link"):
                return [lid for lid in parsed.get("linked_ids", []) if lid in self.memory]
        except Exception as e:
            self._log(f"    [WARN] Link generation JSON parse failed: {e}")
        return []

    # ── Step 3 ────────────────────────────────────────────────────────────────

    def _evolve_neighbours(self, note: MemoryNote, neighbours: list[MemoryNote]):
        r"""
        Equation (7): m*_j ← LLM(m_n ∥ M^n_near \ m_j ∥ m_j ∥ P_s3)
        Evolved notes replace originals in self.memory.
        """
        if not neighbours:
            return
        neighbour_text = "\n".join(
            f"  id={n.id} | context={n.context} | tags={n.tags}"
            for n in neighbours
        )
        prompt = PS3_MEMORY_EVOLUTION.format(
            context=note.context,
            content=note.content,
            keywords=note.keywords,
            neighbours=neighbour_text,
        )
        raw = self._llm(prompt)
        try:
            parsed = _safe_json(raw)
            if not parsed.get("should_evolve"):
                self._log("    no evolution needed")
                return
            for update in parsed.get("updates", []):
                nid = update.get("id")
                if nid not in self.memory:
                    continue
                new_ctx = update.get("new_context", "").strip()
                new_tags = update.get("new_tags", [])
                if new_ctx:
                    self.memory[nid].context = new_ctx
                    self._log(f"    evolved context of {nid}")
                if new_tags:
                    self.memory[nid].tags = new_tags
                    self._log(f"    evolved tags    of {nid}")
                # Re-encode after evolution
                n = self.memory[nid]
                concat = " ".join([n.content] + n.keywords + n.tags + [n.context])
                n.embedding = self.encoder.encode(concat, normalize_embeddings=True)
        except Exception as e:
            self._log(f"    [WARN] Memory evolution JSON parse failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════

st.title("A-MEM: Agentic Memory System")

# API Key input
api_key = st.text_input("Enter your OpenAI API key:", type="password")

if api_key:
    try:
        mem = AMem(openai_api_key=api_key, top_k=5, verbose=True)
        
        st.success("API Key accepted! Memory system initialized.")
        
        # Demo memories
        memories = [
            "I love hiking in the mountains every weekend. My favourite trail is near Lake Tahoe.",
            "I went on a 10-mile trail run last Saturday and felt exhausted but accomplished.",
            "Started learning Python two months ago. I'm building a personal finance tracker.",
            "I enjoy cooking Italian food, especially homemade pasta and risotto.",
            "I read 'Deep Work' by Cal Newport and it changed how I structure my mornings.",
            "I recently upgraded my running shoes to the Brooks Ghost 16 – great for long runs.",
        ]
        
        if st.button("Add Demo Memories"):
            with st.spinner("Adding memories..."):
                for m in memories:
                    mem.add(m)
            st.success("Demo memories added!")
        
        # Show all notes
        if st.button("Show All Stored Notes"):
            notes = mem.show_all()
            st.text_area("All Stored Notes", notes, height=300)
        
        # Retrieval
        query = st.text_input("Enter a query to retrieve memories:")
        if query and st.button("Retrieve"):
            results = mem.retrieve(query, top_k=3)
            for r in results:
                st.write(r.summary())
                st.write("---")
        
        # Chat
        question = st.text_input("Ask a question about the memories:")
        if question and st.button("Chat"):
            answer = mem.chat(question)
            st.write("Answer:", answer)
            
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please enter your OpenAI API key to continue.")