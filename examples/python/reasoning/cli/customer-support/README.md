# CLI Example: Customer Support

A walkthrough showing how to use the `rfnry-rag` CLI to analyze customer support tickets, classify them by type, check agent responses against company policy, analyze multi-turn conversations, and evaluate response quality.

The `customer-support/` folder contains 5 simulated customer support interactions, a multi-turn conversation JSON, a response policy document, and JSON configs for analysis dimensions and ticket categories.

## 1. Install and Configure

```bash
uv add "rfnry-rag[cli]"
rfnry-rag reasoning init
```

Edit `~/.config/rfnry_rag/.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

Edit `~/.config/rfnry_rag/config.toml`:

```toml
[language_model]
provider = "anthropic"
model = "claude-sonnet-4-5-20250929"
```

Suppress verbose BAML logging (optional):

```bash
export BAML_LOG=error
```

## 2. Analyze Tickets

Analyze a ticket to extract intent, urgency, sentiment, and escalation risk:

```bash
rfnry-rag reasoning analyze --file ticket-01-wrong-size.md --summarize --dimensions dimensions.json
```

```
Intent: Customer reporting wrong product received and requesting correct replacement
  with expedited shipping (98% confidence)

Summary: Customer Sarah M. received incorrect air filter sizes (16x20x1 instead of
  ordered 20x25x1) and needs urgent replacement due to running HVAC without filter for
  3 days while having allergies. Agent Mike resolved by shipping correct filters with
  discount applied and letting customer keep wrong items.

Dimensions:
  urgency: 0.85 (90%)
    Customer explicitly states HVAC has been running without filter for 3 days, has
    allergies, and requests ASAP shipping. Health concerns create high time-sensitivity.
  sentiment: frustrated (92%)
    Customer explicitly states 'This is really frustrating' and emphasizes the
    inconvenience of the error. Tone is clearly dissatisfied though not hostile.
  escalation_risk: 0.25 (85%)
    Issue was resolved quickly and favorably (free replacement, kept discount, no return
    needed). Customer's frustration was addressed proactively.
  revenue_impact: low (80%)
    Order involves two air filters with 15% discount. Company absorbed cost of wrong
    shipment but retained customer.
```

Compare with a high-risk ticket:

```bash
rfnry-rag reasoning analyze --file ticket-02-subscription-cancel.md --summarize --dimensions dimensions.json
```

```
Intent: Cancel subscription and request refund for unused product shipment (100% confidence)

Summary: Customer James T. demands immediate cancellation of his $47.99 quarterly filter
  subscription due to unused inventory accumulating in his garage, and requests a refund
  for the March 15th shipment. This is his third attempt to resolve the issue.

Dimensions:
  urgency: 0.95 (95%)
    Customer explicitly demands immediate action ('right now', 'today'), threatens
    credit card dispute if not resolved same day, third failed attempt.
  sentiment: frustrated (100%)
    Multiple indicators: repeated failed attempts, 40-minute hold time, threatening
    chargeback, emphatic language throughout.
  escalation_risk: 0.85 (90%)
    Explicit threat to dispute charges with credit card company, third failed resolution
    attempt, and accumulated frustration.
  revenue_impact: medium (85%)
    Immediate: $47.99 refund + loss of $191.96 annual recurring revenue. Additional
    risk of negative reviews.
```

The `dimensions.json` file defines what to score:

```json
[
  {"name": "urgency", "description": "How time-sensitive is this request", "scale": "0.0-1.0"},
  {"name": "sentiment", "description": "Customer emotional state", "scale": "frustrated/neutral/satisfied"},
  {"name": "escalation_risk", "description": "Likelihood this escalates to a complaint or chargeback", "scale": "0.0-1.0"},
  {"name": "revenue_impact", "description": "Potential revenue at stake (refund, churn, upsell)", "scale": "low/medium/high"}
]
```

## 3. Classify Tickets

Route tickets automatically using category definitions:

```bash
rfnry-rag reasoning classify --file ticket-01-wrong-size.md --categories categories.json
```

```
Category: order_issue (95%)
Strategy: llm
Reasoning: The customer received the wrong size filters (16x20x1 instead of 20x25x1),
  which is a clear case of an incorrect item being shipped.
Runner-up: billing_and_refund (30%)
```

```bash
rfnry-rag reasoning classify --file ticket-03-bulk-pricing.md --categories categories.json
```

```
Category: sales_inquiry (100%)
Strategy: llm
Reasoning: This is a clear commercial inquiry about bulk pricing, recurring orders, and
  setting up a commercial account for a 120-unit apartment complex.
```

Batch-classify all tickets to get a routing summary:

```bash
for ticket in ticket-*.md; do
  name=$(basename "$ticket" .md)
  category=$(rfnry-rag reasoning --json classify --file "$ticket" --categories categories.json | jq -r '.category')
  echo "$name: $category"
done
```

```
ticket-01-wrong-size: order_issue
ticket-02-subscription-cancel: account_management
ticket-03-bulk-pricing: sales_inquiry
ticket-04-defective-product: product_defect
ticket-05-shipping-delay: shipping_delay
```

## 4. Check Responses Against Policy

This is where rfnry-rag becomes a QA tool. Check whether agent responses follow your company's support policy:

```bash
rfnry-rag reasoning compliance --file ticket-01-wrong-size.md --references response-policy.md
```

```
Compliance: FAIL (0.45)

Violations (6):
  [low] tone_and_language — Uses 'Hey Sarah!' as greeting instead of professional opening
    Suggestion: Use 'Hi Sarah,' or 'Hello Sarah,'
  [medium] tone_and_language — Uses prohibited casual intensifier 'totally' in 'I totally get'
    Suggestion: Replace with 'I understand how frustrating that must be'
  [medium] tone_and_language — Uses prohibited slang 'toss them' when referring to items
    Suggestion: Use 'You can dispose of the incorrect filters or donate them'
  [medium] tone_and_language — Uses prohibited slang 'holler' in closing
    Suggestion: Replace with 'please don't hesitate to reach out'
  [low] tone_and_language — Does not acknowledge the specific issue before offering solutions
    Suggestion: Add 'I see that you received 16x20x1 filters instead of the 20x25x1'
  [medium] shipping_issues — States 'you should have them in 2-3 business days' which
    implies a promised delivery date
    Suggestion: Use 'They are expected to arrive within 2-3 business days'
```

Mike's response is friendly but violates tone guidelines — too casual. Now check a more serious case:

```bash
rfnry-rag reasoning compliance --file ticket-04-defective-product.md --references response-policy.md
```

```
Compliance: FAIL (0.55)

Violations (5):
  [high] damage_claims — Agent stated 'I've submitted your $185 reimbursement request'
    which violates policy to say 'we will review' not 'I've submitted'. This creates an
    expectation of approval.
    Suggestion: Change to 'I'll need to submit your $185 cleaning cost to our claims
    team for review.'
  [high] damage_claims — The $185 reimbursement exceeds the $100 maximum without
    escalation, and requires supervisor approval. Agent should not have implied this
    would be processed without supervisor involvement.
    Suggestion: Add: 'Since this amount exceeds our standard threshold, a supervisor
    will need to review and approve the claim.'
  [medium] damage_claims — Agent requested the invoice after stating the claim was
    submitted, but policy requires collecting invoice AND photos before submitting.
    Suggestion: Request both upfront before mentioning submission.
  [medium] shipping_issues — Agent promised 'You should receive it by Monday' which
    violates the policy to never promise specific delivery dates.
    Suggestion: Change to 'It's expected to arrive by Monday.'
  [low] damage_claims — 'These claims are typically approved within 5 business days'
    creates expectation of approval when policy says not to guarantee.
    Suggestion: Change to 'The review process typically takes about 5 business days.'
```

Dana's response has real policy violations — she promised $185 reimbursement that exceeds her authority ($100 max without escalation) and guaranteed approval she can't confirm.

Use `--threshold` to create an automated compliance gate. With threshold 0.7, a score of 0.55 fails:

```bash
rfnry-rag reasoning compliance --file ticket-04-defective-product.md --references response-policy.md --threshold 0.7
```

```
Compliance: FAIL (0.55)

Violations (4):
  [high] damage_claims — Agent stated 'I've submitted your $185 reimbursement request'
    which violates the policy that agents should not say 'I've submitted' but rather
    'we will review'.
    Suggestion: Replace 'I've submitted your $185 reimbursement request to our quality
    team' with 'We will review your $185 reimbursement request for the HVAC cleaning cost'
  [high] damage_claims — The $185 reimbursement amount exceeds the $100 maximum without
    escalation. The agent should have obtained supervisor approval before proceeding.
    Suggestion: The agent should escalate to a supervisor for approval before responding.
  [medium] damage_claims — 'These claims are typically approved within 5 business days'
    implies approval is likely, violating the spirit of not guaranteeing reimbursement.
    Suggestion: Remove the statement about typical approval and instead say: 'Once we
    receive the invoice, we'll review your claim and follow up with our decision.'
  [medium] shipping_issues — Agent promised 'You should receive it by Monday' for overnight
    shipping, violating the policy to say 'expected by' not 'will arrive by'.
    Suggestion: Change to 'It's expected to arrive by Monday'
```

An AI support agent can use this as a quality gate: above threshold is auto-approve, below is human review.

## 5. Analyze Conversations

For multi-turn support threads, `analyze-context` tracks how intent shifts, detects escalation, and determines resolution status across the full exchange:

```bash
rfnry-rag reasoning analyze-context --file conversation-ticket-02.json --summarize --dimensions dimensions.json
```

```
Intent: Cancel subscription and obtain refund (98% confidence)

Summary: Customer James T. successfully cancels a quarterly filter subscription ($47.99)
  after multiple failed attempts, expressing frustration about unused products accumulating.
  Agent processes immediate cancellation, approves refund for last shipment, and escalates
  website cancellation issue to technical team.

Dimensions:
  urgency: 0.85 (90%)
    Customer explicitly threatens credit card dispute if not resolved today, has made three
    prior attempts to resolve, and demands immediate action with emphatic language.
  sentiment: frustrated (95%)
    Customer uses emphatic capitalization, mentions three failed resolution attempts
    (40-minute hold, unhelpful automated email), and threatens chargeback dispute.
  escalation_risk: 0.75 (85%)
    Customer explicitly threatens to dispute all charges with credit card company. Multiple
    prior failed contact attempts indicate pattern of unresolved issues. Risk reduced by
    agent's immediate action and refund approval.
  revenue_impact: medium (90%)
    Immediate revenue loss of $47.99 refund plus $191.96 annual recurring revenue from
    cancelled quarterly subscription. Potential additional chargeback fees.

Intent Shifts:
  [6] Cancel subscription → Cancel subscription and obtain refund
    After cancellation is confirmed, customer adds new demand for refund of last shipment
    and escalates with chargeback threat, expanding scope beyond simple cancellation.

Escalation: yes
  Multiple escalation signals: explicit threat to dispute charges, third attempt to resolve
  issue after two failed contacts, emphatic language, ultimatum with same-day deadline.
  However, agent's immediate action likely prevented actual escalation.
Resolution: resolved
```

The `conversation-ticket-02.json` is a JSON array of `{role, text}` objects representing each message in the exchange:

```json
[
  {"role": "customer", "text": "I want to cancel my filter subscription..."},
  {"role": "agent", "text": "Thank you for reaching out..."},
  {"role": "customer", "text": "I don't have my subscription ID handy..."},
  ...
]
```

The key insight: single-ticket analysis (section 2) tells you about urgency and sentiment. Context analysis tells you *how the conversation evolved* — where intent shifted, whether escalation was detected, and whether the issue was resolved.

## 6. Evaluate Response Quality

Compare the quality of two different agent responses:

```bash
rfnry-rag reasoning evaluate \
  --generated ticket-01-wrong-size.md \
  --reference ticket-02-subscription-cancel.md \
  --strategy judge
```

```
Score: 0.75 (medium)
Judge: 0.75
  The generated output demonstrates good customer service fundamentals with prompt
  problem resolution, clear communication, and appropriate tone. However, it addresses
  a simpler issue compared to the reference's more complex scenario involving
  cancellation difficulties, multiple contact attempts, and escalation threats.

Dimension Scores:
  problem_resolution: 0.85
  tone_and_professionalism: 0.70
  completeness: 0.80
  empathy_and_acknowledgment: 0.65
  clarity: 0.85
  proactive_communication: 0.70
```

## 7. Piped Input for Automation

AI models or scripts can pipe text directly and get JSON back:

```bash
echo "I've been waiting 2 weeks for my order and nobody will help me. \
This is unacceptable. I want a full refund NOW." \
  | rfnry-rag reasoning --json analyze --dimensions dimensions.json
```

```json
{
  "primary_intent": "Demand refund for severely delayed order",
  "confidence": 0.97,
  "dimensions": {
    "urgency": {"name": "urgency", "value": "0.95", "confidence": 0.92},
    "sentiment": {"name": "sentiment", "value": "frustrated", "confidence": 0.98},
    "escalation_risk": {"name": "escalation_risk", "value": "0.90", "confidence": 0.88},
    "revenue_impact": {"name": "revenue_impact", "value": "medium", "confidence": 0.80}
  }
}
```

## 8. A QA Workflow

A support team lead reviewing agent performance:

```bash
# 1. Classify all tickets to check routing accuracy
for ticket in ticket-*.md; do
  echo "$(basename $ticket): $(rfnry-rag reasoning --json classify --file $ticket --categories categories.json | jq -r .category)"
done

# 2. Check every response against policy
for ticket in ticket-*.md; do
  score=$(rfnry-rag reasoning --json compliance --file "$ticket" --references response-policy.md | jq '.score')
  echo "$(basename $ticket): compliance $score"
done

# 3. Flag high-urgency tickets that might need follow-up
for ticket in ticket-*.md; do
  urgency=$(rfnry-rag reasoning --json analyze --file "$ticket" --dimensions dimensions.json | jq -r '.dimensions.urgency.value')
  echo "$(basename $ticket): urgency $urgency"
done
```

This gives you a dashboard of your support quality without writing any code.
