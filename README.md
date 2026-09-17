# Education AI - Cote d'Ivoire

Education AI is a curriculum-grounded conversational education prototype for
Cote d'Ivoire. It combines retrieval-augmented generation (RAG) with an
educational generation and quality-control pipeline. The architecture is
designed to be adaptable to other African education systems while keeping each
deployment's curriculum, terminology, sources, and governance separate.

## Validated mini-MVP

The current vertical slice has been validated end to end:

1. An educational request is embedded with OpenAI `text-embedding-3-small`.
2. Supabase retrieves relevant educational context from the existing AskCI
   vector index.
3. Mistral `ministral-3b-latest` generates the educational content.
4. The existing OpenAI quality check evaluates the generated result.

The validated demonstration generated a college-level mathematics exercise on
proportionality from retrieved educational context. The quality check accepted
the result with a score of 95/100.

Mistral is the primary text-generation layer. Existing OpenAI embeddings are
temporarily retained for compatibility with the current vector index; this MVP
does not migrate or re-index the corpus. Supabase currently provides the data
and vector-retrieval layer.

## Scope

This repository is an accelerator prototype, not a production deployment. It
does not claim fine-tuning, nationwide deployment, government endorsement,
complete infrastructure sovereignty, or large-scale user validation.

Private educational documents, local environment files, generated subjects,
and credentials are not distributed in this repository.

## Project structure

```text
src/
  education_engine/
    ai_providers.py
    generate_similar_subject.py
    askci_subject_service.py
    education_subjects_api.py
tests/
  test_mistral_integration.py
.env.example
```

## Configuration

Create `src/education_engine/.env` from `.env.example` and provide your own
credentials. Never commit the resulting `.env` file.

Required services and variables:

- Mistral generation: `MISTRAL_API_KEY`
- Existing OpenAI embeddings and quality check: `OPENAI_API_KEY`
- Supabase retrieval: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`

The default verified generation model is `ministral-3b-latest`.

## Local verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The smoke tests validate provider selection, Mistral response handling, and the
configured OpenAI fallback without making network requests.

## Security and data

No API keys, `.env` files, private document banks, or generated educational
outputs should be committed. The repository contains source code and safe
configuration placeholders only.
