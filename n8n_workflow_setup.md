# Credit Risk n8n Setup

Use these node names so the expressions below work as written:

- `Webhook`
- `Predict API`
- `Ollama`
- `Needs Manual Review`
- `Review Summary API`
- `Prepare Audit Record`
- `Slack Summary`
- `Google Sheets`

## 1. Webhook

Send a `POST` request to your n8n webhook with a full borrower payload.

Example:

```json
{
  "credit_amount": 5000,
  "duration": 24,
  "age": 35,
  "status_of_existing_checking_account": "A12",
  "credit_history": "A32",
  "purpose": "A43",
  "savings_account_bonds": "A61",
  "present_employment_since": "A73",
  "installment_rate_in_percentage_of_disposable_income": 2,
  "personal_status_and_sex": "A93",
  "other_debtors_guarantors": "A101",
  "present_residence_since": 2,
  "property": "A121",
  "other_installment_plans": "A143",
  "housing": "A152",
  "number_of_existing_credits_at_this_bank": 1,
  "job": "A173",
  "number_of_people_being_liable_to_provide_maintenance_for": 1,
  "telephone": "A191",
  "foreign_worker": "A201"
}
```

The API accepts both `duration` and `duration_in_month`, and both `age` and `age_in_years`.

## 2. Predict API

- Method: `POST`
- URL: `http://localhost:8000/predict`
- Send Body: `true`
- Body Content Type: `JSON`

Body:

```json
{
  "credit_amount": {{$json["credit_amount"]}},
  "duration": {{$json["duration"] ?? $json["duration_in_month"]}},
  "age": {{$json["age"] ?? $json["age_in_years"]}},
  "status_of_existing_checking_account": "{{$json["status_of_existing_checking_account"]}}",
  "credit_history": "{{$json["credit_history"]}}",
  "purpose": "{{$json["purpose"]}}",
  "savings_account_bonds": "{{$json["savings_account_bonds"]}}",
  "present_employment_since": "{{$json["present_employment_since"]}}",
  "installment_rate_in_percentage_of_disposable_income": {{$json["installment_rate_in_percentage_of_disposable_income"]}},
  "personal_status_and_sex": "{{$json["personal_status_and_sex"]}}",
  "other_debtors_guarantors": "{{$json["other_debtors_guarantors"]}}",
  "present_residence_since": {{$json["present_residence_since"]}},
  "property": "{{$json["property"]}}",
  "other_installment_plans": "{{$json["other_installment_plans"]}}",
  "housing": "{{$json["housing"]}}",
  "number_of_existing_credits_at_this_bank": {{$json["number_of_existing_credits_at_this_bank"]}},
  "job": "{{$json["job"]}}",
  "number_of_people_being_liable_to_provide_maintenance_for": {{$json["number_of_people_being_liable_to_provide_maintenance_for"]}},
  "telephone": "{{$json["telephone"]}}",
  "foreign_worker": "{{$json["foreign_worker"]}}"
}
```

## 3. Ollama

- Method: `POST`
- URL: `http://localhost:11434/api/generate`
- Send Body: `true`
- Body Content Type: `JSON`

Body:

```json
{
  "model": "gemma3:4b",
  "stream": false,
  "prompt": "Explain this credit decision clearly:\nPD: {{$json[\"pd\"]}}\nDecision: {{$json[\"decision\"]}}\nReason: {{$json[\"decision_reason\"]}}"
}
```

Ollama returns the text in:

```text
{{$json["response"]}}
```

## 4. Needs Manual Review

Add an `If` node after `Ollama`. Test the value:

```text
{{$items("Predict API", 0, 0)[0].json["review_required"]}}
```

Route `true` to `Review Summary API`, and route `false` directly to
`Prepare Audit Record`.

## 5. Review Summary API

- Method: `POST`
- URL: `http://localhost:8000/review-summary`
- Send Body: `true`
- Body Content Type: `JSON`

Body:

```javascript
={{ {
  applicant: $items("Webhook", 0, 0)[0].json.body,
  prediction: $items("Predict API", 0, 0)[0].json
} }}
```

The API uses Chroma retrieval over local policy documents only for applications
whose prediction is `Review`.

## 6. Slack

Use explicit node references because the current JSON comes from the Ollama node.

Message:

```text
Loan Decision

Decision: {{$node["Predict API"].json["decision"]}}
PD: {{$node["Predict API"].json["pd"]}}
Explanation: {{$node["Ollama"].json["response"]}}
Expected Loss: {{$node["Predict API"].json["applicant_expected_loss"]}}
```

## 7. Google Sheets

Append row fields:

- `timestamp` -> `{{$now}}`
- `pd` -> `{{$node["Predict API"].json["pd"]}}`
- `decision` -> `{{$node["Predict API"].json["decision"]}}`
- `explanation` -> `{{$node["Ollama"].json["response"]}}`
- `expected_loss` -> `{{$node["Predict API"].json["applicant_expected_loss"]}}`
- `policy_version` -> `{{$node["Predict API"].json["policy_version"]}}`
- `policy_constraints_met` -> `{{$node["Predict API"].json["policy_constraints_met"]}}`
- `policy_selection_reason` -> `{{$node["Predict API"].json["policy_selection_reason"]}}`
- `review_required` -> `{{$node["Predict API"].json["review_required"]}}`
- `review_summary` -> the prepared audit record field of the same name
- `retrieved_policy_sources` -> the prepared audit record field of the same name
- `embedding_model` -> the prepared audit record field of the same name
- `retrieval_policy_version` -> the prepared audit record field of the same name
- `final_human_action` -> blank until completed by a reviewer

## Common fixes

- `422` from FastAPI: a required field is missing or a category code is invalid.
- Blank Slack fields: use `$node["Predict API"]...` instead of `$json[...]`.
- Blank Ollama explanation: make sure `stream` is `false` and read `response`.
- Empty `review_summary`: this is expected for `Approve` and `Reject` cases; for a `Review` case confirm Ollama has both `gemma3:4b` and `nomic-embed-text`.
- Google Sheets blanks: map from `Predict API` and `Ollama` explicitly, not from the current node by accident.
