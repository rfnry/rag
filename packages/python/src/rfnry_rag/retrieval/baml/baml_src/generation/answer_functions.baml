function GenerateAnswer(system_prompt: string, context: string, query: string, history: string) -> string {
  client Default
  prompt #"
    {{ _.role("system") }}
    {{ system_prompt }}

    {% if history %}
    {{ history }}
    {% endif %}

    {{ _.role("user") }}
    Context:
    {{ context }}

    Question: {{ query }}
  "#
}
