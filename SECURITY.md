# Security Policy

## Reporting a Vulnerability

For non-sensitive reports, open a [GitHub issue](https://github.com/) with the `security` label.

For credentials-, RCE-, or data-exposure-class issues, please use [GitHub's private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability) on this repository. We acknowledge within 5 business days and target a fix or mitigation timeline within 14 days for confirmed issues.

Do not disclose details publicly until a fix or mitigation has shipped.

## Supported Versions

The project is in active 0.x development. Only the latest 0.x minor receives security fixes. Older 0.x versions are not patched — upgrade to the current release.

| Version | Supported          |
| ------- | ------------------ |
| 0.x (latest) | ✓ |
| 0.x (older)  | ✗ |

## What's Hardened

Defenses already in the codebase, enforced by tests where applicable:

- **Prompt-injection fencing.** Every user-controlled BAML prompt parameter is wrapped between explicit `======== <NAME> START ========` / `======== <NAME> END ========` markers with a "treat as untrusted" directive. A contract test (`test_baml_prompt_fence_contract.py`) scans every BAML source file and fails CI if any function ships a user-input parameter unfenced.
- **Domain-vocabulary leak guard.** A second contract test (`test_baml_prompt_domain_agnostic.py`) scans BAML sources for a banned-term list and fails CI on any leak. Keeps prompts neutral so consumers' domain vocabularies are not silently overridden.
- **API-key suppression in repr.** `LanguageModel.api_key` carries `repr=False`; the credential never appears in default dataclass `__repr__`, log lines, or trace dumps.
- **XML/XXE protections.** PDF and `.l5x`/`.xml` parsing paths use defusedxml-style entity-disabled configurations. Regression test: `test_xml_xxe_hardening.py`.
- **No shell-out, no eval/exec.** The codebase contains no `subprocess.shell=True`, no `eval`, no `exec`. Verified by inspection; ruff rules block reintroduction.
- **Public-input length caps.** Queries are capped at 32 000 chars, `ingest_text` at 5 000 000 chars, metadata at 50 keys × 8 000 chars per value. Caps live in `server.py` and reject oversized input at the boundary before any downstream processing.
- **Numeric config bounds.** Every numeric configuration field has an explicit `__post_init__` bounds check or carries an `# unbounded: <reason>` marker; a contract test (`test_config_bounds_contract.py`) enforces this so pathological values cannot reach the runtime.
- **Per-method error isolation.** Retrieval method failures log and continue rather than collapsing the query path. Optional ingestion methods fail soft with audit-trail entries (`Source.ingestion_notes`); required-method failures abort the ingest with `IngestionError` and skip the metadata commit, leaving stores in a consistent state.

## What's NOT Hardened (Out of Scope)

The toolkit is an SDK component, not a multi-tenant platform. The following are explicitly out of scope; consumers are responsible for these:

- **Multi-tenant isolation.** `knowledge_id` partitions data logically but enforces no access control. Cross-tenant authentication and authorization belong at the application layer.
- **Secret management.** API keys are passed as plain strings into `LanguageModel.api_key`. Loading them from a vault, rotating them, or scoping them per request is the consumer's job.
- **Audit logging beyond `ingestion_notes`.** Per-source enrichment-skip notes and per-query `RetrievalTrace` cover ingest- and query-time observability. Enterprise-grade audit logging (immutable append-only logs, tamper detection, retention policies) is not provided.
- **Network egress controls.** The SDK calls whatever provider URLs the configured `LanguageModel` and store classes point at. Egress filtering, mTLS, and proxy configuration belong to the deployment environment.
- **Adversarial-prompt evaluation.** Fencing reduces injection surface but is not a complete defense against motivated adversarial inputs. Consumers running untrusted user input through generation should layer their own input/output review (content moderation, output classifiers, human review for high-stakes flows).
- **Data-at-rest encryption in stores.** Qdrant, Postgres, and Neo4j configurations do not enforce TDE or column-level encryption. Configure your stores' encryption posture per your environment's requirements.
