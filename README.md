# Education AI Côte d'Ivoire

**A curriculum-grounded conversational AI prototype for African education
systems, starting with Côte d'Ivoire and powered by Mistral.**

## The problem

Students and educators increasingly have access to general-purpose AI systems,
but these systems are not inherently grounded in Côte d'Ivoire's educational
programmes, pedagogical terminology, curriculum structure, or locally relevant
educational resources.

This creates risks such as:

- answers disconnected from the curriculum;
- hallucinated or unsuitable educational content;
- limited traceability to trusted educational sources;
- dependence on external general-purpose AI infrastructure.

## Our solution

Education AI Côte d'Ivoire is building a conversational educational AI layer
grounded in curriculum-specific knowledge. The prototype combines:

- retrieval-augmented generation;
- educational knowledge and context retrieval;
- Mistral text generation;
- source-grounded educational content;
- quality-control mechanisms.

Côte d'Ivoire is the initial implementation. The architecture separates the AI
engine from country- and curriculum-specific knowledge so it can progressively
be adapted to other African education systems.

## Why Mistral

Mistral's efficient models and open-weight ecosystem align with the project's
long-term objective of reducing dependency on fully proprietary AI
infrastructure and progressively increasing control over model deployment and
educational data infrastructure.

The current mini-MVP still uses external APIs and is not a fully sovereign
deployment.

## Validated mini-MVP

The following end-to-end flow has been technically validated:

```text
Educational request
        |
        v
OpenAI text-embedding-3-small
        |
        v
Supabase curriculum and educational retrieval
        |
        v
Relevant educational context
        |
        v
Mistral ministral-3b-latest generation
        |
        v
OpenAI quality control
        |
        v
Grounded educational output
```

Validated case:

- Subject: Mathematics
- Education level: Collège
- Class: 3e
- Theme: Proportionality
- Request: Exercise generation without correction
- Retrieval: 6 programme sources and 4 model/context sources
- Mistral generation: successful
- OpenAI generation fallback: not used
- Quality check: 95/100

Existing OpenAI embeddings are temporarily retained for compatibility with the
current vector index. The mini-MVP does not migrate or re-index the corpus, and
private corpus content is not included in this repository.

## African scalability

The architecture is intended to support separate knowledge packs and
configurations by:

- country;
- curriculum;
- education level;
- subject;
- language.

The initial target is Côte d'Ivoire in French. Future work may adapt the system
to other African education systems and progressively support African and local
languages. These adaptations are a design direction, not deployed
functionality.

## Current status

**Prototype / validated mini-MVP**

Validated:

- Mistral API integration;
- RAG retrieval;
- end-to-end educational generation;
- provider selection;
- quality-control pipeline.

Not yet claimed:

- fine-tuning;
- production readiness;
- nationwide deployment;
- government endorsement;
- complete sovereign infrastructure;
- large-scale user validation.

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

The verified default generation model is `ministral-3b-latest`.

## Local verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The smoke tests validate provider selection, Mistral response handling, and the
configured OpenAI fallback without making network requests.

## Data and security

The public repository intentionally excludes:

- API keys and credentials;
- `.env` files;
- private or local educational corpus files;
- generated educational outputs.

Educational sources should be described according to their individually
verified provenance; the project does not assume that every corpus document is
official.

## Vision

The long-term objective is to create trustworthy, curriculum-aware AI
infrastructure that can be adapted to African education systems while
progressively increasing local control over models, knowledge, deployment, and
educational data.
