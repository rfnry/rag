retry_policy Retry1 {
  max_retries 1
  strategy {
    type exponential_backoff
  }
}

retry_policy Retry2 {
  max_retries 2
  strategy {
    type exponential_backoff
  }
}

retry_policy Retry3 {
  max_retries 3
  strategy {
    type exponential_backoff
  }
}

retry_policy Retry4 {
  max_retries 4
  strategy {
    type exponential_backoff
  }
}

retry_policy Retry5 {
  max_retries 5
  strategy {
    type exponential_backoff
  }
}

// Fallback client used only if no ClientRegistry override is supplied.
// The model name is a sentinel so any unrouted call fails fast with a clear
// provider error instead of silently proceeding on a specific model that may
// be deprecated. Production code must always pass `client_registry` built via
// `build_registry(LLMClient)`.
client<llm> Default {
  provider openai
  retry_policy Retry3
  options {
    model "UNCONFIGURED-route-through-build-registry"
  }
}
