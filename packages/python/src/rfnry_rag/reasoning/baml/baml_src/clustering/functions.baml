function LabelCluster(samples: string) -> ClusterLabel {
  client Default
  prompt #"
    Generate a short descriptive label (2-5 words) for a cluster of related texts.

    ======== SAMPLE TEXTS ========
    {{ samples }}
    ======== END SAMPLES ========

    The label should capture the common theme across these samples.

    {{ ctx.output_format }}
  "#
}
