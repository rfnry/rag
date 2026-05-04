class ExtractedMemoryFact {
  text string @description("Atomic factual statement that should be remembered. Self-contained, single fact, present tense unless an explicit time anchors it.")
  attributed_to string? @description("Role label of the source turn that produced this fact, copied verbatim from the [role] prefix. Null if no role label is visible.")
  linked_memory_row_ids string[] @description("IDs of provided existing memories that this new fact closely relates to. Empty if no relation.")
}

class ExtractedMemoryList {
  memories ExtractedMemoryFact[] @description("Atomic facts extracted from the interaction. Empty if nothing factual is worth remembering.")
}
