# Scope

## Purpose

This project builds a Retrieval-Augmented Generation (RAG) system that answers questions using a supplied set of Indian gender budget documents. The goal is to give short, accurate answers that can be traced back to the source document and page.

## Corpus

| Item | Details |
|---|---|
| Documents | 11 government documents |
| Governments | Government of India, Bihar, Delhi, Odisha |
| Budget years | Multiple years |
| Formats | PDFs, one CSV file, one XLS spreadsheet |
| Languages | English and bilingual Hindi/English |

## What Is In Scope

**Ingestion.** Text is extracted from PDF pages, and rows are extracted from the CSV and XLS files. Each record keeps its source file, page number, row number, and sheet name. The records are then split into overlapping chunks.

**Retrieval.** Two search methods are used together:

- **FAISS** for meaning-based search, using the `all-MiniLM-L6-v2` embedding model.
- **BM25** for keyword search, which helps with exact names and numbers.

The results from both are merged and reranked with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). This gives a wide set of candidates first and then keeps only the most relevant ones for the LLM.

**Multilingual questions.** English questions go directly to retrieval. Non-English questions are translated to English first, so the same English models can be reused.

**Generation.** A hosted LLM (Gemini) writes the answer using only the retrieved passages. Source details are kept with each passage, so answers can include document and page references.

**Refusing to guess.** If the retrieved evidence is not enough, the system does not answer.

## What Is Out of Scope

| Not included | Reason |
|---|---|
| Full cleaning of the corpus | Documents were indexed as extracted, within the assignment time |
| Fixing legacy Hindi font encoding | Some bilingual PDFs produce broken Hindi text; the English part is still usable |
| Rebuilding tables into a database | Tables are treated as text and rows |
| OCR | No separate OCR stage was added |
| Outside knowledge | Answers come only from the supplied documents |
