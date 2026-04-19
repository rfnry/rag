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

client<llm> Default {
  provider openai
  retry_policy Retry3
  options {
    model "gpt-4o-mini"
  }
}
