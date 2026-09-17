\# Education AI - Côte d'Ivoire



Sovereign conversational AI for education, grounded in official curricula and pedagogical resources, initially designed for Côte d'Ivoire and adaptable to other African education systems.



\## Status



\*\*Prototype Foundation — Work in Progress\*\*



This repository documents and implements the initial prototype of Education AI, an AI-powered educational assistant designed around African education systems.



\## Problem



Students and educators in Côte d'Ivoire increasingly have access to general-purpose generative AI, but these systems are not specifically grounded in the official national curriculum, approved pedagogical resources, local educational terminology, or the realities of the Ivorian education system.



This can result in:



\- answers that do not follow the official curriculum;

\- explanations inappropriate for the learner's academic level;

\- inaccurate or difficult-to-verify educational information;

\- dependence on external proprietary AI infrastructure;

\- accessibility limitations in low-connectivity environments.



\## Solution



Education AI is a conversational educational platform combining efficient AI models with a controlled educational knowledge base.



The system is designed to provide answers grounded in trusted educational resources while taking into account:



\- country;

\- curriculum;

\- academic level;

\- subject;

\- pedagogical context;

\- source documents.



Côte d'Ivoire is the initial deployment environment.



\## Mistral AI Strategy



The prototype is designed to explore Mistral open-weight models as its core generative AI layer.



The initial technical approach combines:



1\. Mistral models;

2\. Retrieval-Augmented Generation (RAG);

3\. structured educational document ingestion;

4\. curriculum-aware retrieval;

5\. source-grounded generation;

6\. educational evaluation and guardrails.



Fine-tuning or model adaptation will be evaluated where RAG and prompting alone are insufficient.



\## Planned Architecture



```text

Official Educational Resources

&#x20;           |

&#x20;           v

&#x20;   Document Ingestion

&#x20;           |

&#x20;           v

&#x20;Cleaning \& Classification

&#x20;           |

&#x20;           v

&#x20;  Curriculum Metadata

&#x20;           |

&#x20;           v

&#x20;Semantic Chunking / Index

&#x20;           |

&#x20;           v

&#x20;         RAG

&#x20;           |

&#x20;           v

&#x20;    Mistral Model

&#x20;           |

&#x20;           v

&#x20;Educational Guardrails

&#x20;           |

&#x20;           v

Grounded Answer + Sources

&#x20;           |

&#x20;           v

&#x20;     Web / Mobile UI



Local African Context



The first implementation focuses on Côte d'Ivoire.



The knowledge layer will progressively incorporate legitimate educational resources such as official curricula, pedagogical references and other authorized educational materials.



French will be the initial primary language, with the architecture designed for progressive support of African languages.



Efficient models and modular deployment will also be explored for environments where connectivity, compute capacity or infrastructure is constrained.



Data Strategy



Educational documents will be processed through a structured pipeline:



Source

&#x20; -> Extraction

&#x20; -> Cleaning

&#x20; -> Classification

&#x20; -> Curriculum Metadata

&#x20; -> Semantic Chunking

&#x20; -> Embeddings / Indexing

&#x20; -> Retrieval

&#x20; -> Grounded Generation

&#x20; -> Evaluation



No restricted educational datasets are distributed through this repository.



Initial Prototype Scope



The first prototype will deliberately use a limited educational corpus to validate the approach.



Planned capabilities include:



natural-language educational questions;

curriculum-grounded answers;

grade and subject-aware retrieval;

supporting source references;

learner and educator interaction modes;

evaluation of factuality and curriculum alignment.

Evaluation



The prototype will progressively measure:



retrieval relevance;

factual correctness;

curriculum alignment;

hallucination rate;

source traceability;

grade appropriateness;

latency and resource requirements.

Technology Direction



Initial technology direction:



Mistral open-weight models

Python

RAG

Vector search

FastAPI

Next.js

TypeScript

PostgreSQL

Containerized deployment



The final technical choices will be validated through experimentation.



Scalability



The platform separates the AI engine from country-specific educational knowledge.



This architecture is intended to enable:



Côte d'Ivoire

&#x20;     |

&#x20;     v

Country-specific Education Knowledge Packs

&#x20;     |

&#x20;     v

Francophone West Africa

&#x20;     |

&#x20;     v

Other African Education Systems



Each deployment can maintain its own curriculum, terminology, sources and governance requirements.



Roadmap

Phase 1 — Foundation



Architecture, corpus definition, data structure and evaluation methodology.



Phase 2 — Functional Prototype



Mistral integration, document ingestion, retrieval and conversational API.



Phase 3 — Validation



Testing with educators and learners and measuring curriculum alignment and hallucinations.



Phase 4 — MVP



Production web interface, user management, educator/learner modes, monitoring and scalable deployment.



Vision



Education AI aims to demonstrate that educational AI for Africa can be built around local curricula, controlled knowledge, measurable educational quality and open AI infrastructure.



Built from Côte d'Ivoire. Designed to scale across Africa.





\### 3. Vérifions avant le commit



Une fois Notepad fermé :



```powershell

git status

