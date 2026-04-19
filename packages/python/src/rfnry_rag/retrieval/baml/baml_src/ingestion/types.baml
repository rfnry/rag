class PageAnalysis {
  description string
  entities DiscoveredEntity[]
  tables DiscoveredTable[]
  annotations string[]
  page_type string
}

class DiscoveredEntity {
  name string
  category string
  value string?
  context string
}

class DiscoveredTable {
  title string?
  columns string[]
  rows map<string, string>[]
}