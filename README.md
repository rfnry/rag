# rfnry-knowledge

A modular, **provider-agnostic** retrieval engine for Python. Compose vector, document, and graph retrieval into one pipeline, fuse the results, and route between indexed retrieval and full-context generation based on corpus size — automatically. The engine ships zero provider implementations; the host application brings any LLM, embedder, or reranker that conforms to the library's Protocols and plugs it in. Built around a single principle: as language models grow stronger and contexts grow longer, the toolkit gets out of their way instead of working around them.

See [`packages/python/README.md`](packages/python/README.md) for the SDK reference.

## License

MIT — see [LICENSE](LICENSE).
