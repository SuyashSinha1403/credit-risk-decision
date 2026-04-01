from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import shap
from fastapi import FastAPI
from joblib import dump, load
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_PATH = DATA_DIR / "german.data"
ARTIFACTS_PATH = ARTIFACTS_DIR / "credit_risk_artifacts.joblib"
DATA_COLUMNS = [
    "status of existing checking account",
    "duration in month",
    "credit history",
    "purpose",
    "credit amount",
    "savings account/bonds",
    "present employment since",
    "installment rate in percentage of disposable income",
    "personal status and sex",
    "other debtors / guarantors",
    "present residence since",
    "property",
    "age in years",
    "other installment plans",
    "housing",
    "number of existing credits at this bank",
    "job",
    "number of people being liable to provide maintenance for",
    "telephone",
    "foreign worker",
    "target",
]

THRESHOLD_SETS = [
    (0.15, 0.35),
    (0.20, 0.40),
    (0.25, 0.50),
]
LGD = 0.4
REVIEW_BOOKING_RATE = 0.5
MIN_APPROVAL_RATE = 0.40
MAX_MISSED_DEFAULTS = 5

FEATURE_LABEL_MAP = {
    "status_of_existing_checking_account_A11": "checking account below 0 DM",
    "status_of_existing_checking_account_A12": "checking account between 0 and 200 DM",
    "status_of_existing_checking_account_A13": "checking account above 200 DM",
    "status_of_existing_checking_account_A14": "no checking account",
    "credit_history_A30": "no credits taken",
    "credit_history_A31": "all credits paid back duly",
    "credit_history_A32": "existing credits paid back duly",
    "credit_history_A33": "delay in paying off credits",
    "credit_history_A34": "critical credit history",
    "savings_account/bonds_A61": "low savings",
    "savings_account/bonds_A62": "moderate savings",
    "savings_account/bonds_A63": "higher savings",
    "savings_account/bonds_A64": "very high savings",
    "savings_account/bonds_A65": "unknown or no savings",
    "duration_in_month": "longer loan duration",
    "credit_amount": "higher credit amount",
    "installment_rate_in_percentage_of_disposable_income": "higher installment burden",
    "age_in_years": "older borrower age",
}

RAW_TO_API_FIELD = {
    "status_of_existing_checking_account": "status_of_existing_checking_account",
    "duration_in_month": "duration_in_month",
    "credit_history": "credit_history",
    "purpose": "purpose",
    "credit_amount": "credit_amount",
    "savings_account/bonds": "savings_account_bonds",
    "present_employment_since": "present_employment_since",
    "installment_rate_in_percentage_of_disposable_income": "installment_rate_in_percentage_of_disposable_income",
    "personal_status_and_sex": "personal_status_and_sex",
    "other_debtors_/_guarantors": "other_debtors_guarantors",
    "present_residence_since": "present_residence_since",
    "property": "property",
    "age_in_years": "age_in_years",
    "other_installment_plans": "other_installment_plans",
    "housing": "housing",
    "number_of_existing_credits_at_this_bank": "number_of_existing_credits_at_this_bank",
    "job": "job",
    "number_of_people_being_liable_to_provide_maintenance_for": "number_of_people_being_liable_to_provide_maintenance_for",
    "telephone": "telephone",
    "foreign_worker": "foreign_worker",
}


class BorrowerInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "status_of_existing_checking_account": "A12",
                "duration": 24,
                "credit_history": "A32",
                "purpose": "A43",
                "credit_amount": 5000,
                "savings_account_bonds": "A61",
                "present_employment_since": "A73",
                "installment_rate_in_percentage_of_disposable_income": 2,
                "personal_status_and_sex": "A93",
                "other_debtors_guarantors": "A101",
                "present_residence_since": 2,
                "property": "A121",
                "age": 35,
                "other_installment_plans": "A143",
                "housing": "A152",
                "number_of_existing_credits_at_this_bank": 1,
                "job": "A173",
                "number_of_people_being_liable_to_provide_maintenance_for": 1,
                "telephone": "A191",
                "foreign_worker": "A201",
            }
        },
    )

    status_of_existing_checking_account: Literal["A11", "A12", "A13", "A14"]
    duration_in_month: int = Field(validation_alias=AliasChoices("duration_in_month", "duration"))
    credit_history: Literal["A30", "A31", "A32", "A33", "A34"]
    purpose: Literal["A40", "A41", "A410", "A42", "A43", "A44", "A45", "A46", "A48", "A49"]
    credit_amount: float
    savings_account_bonds: Literal["A61", "A62", "A63", "A64", "A65"]
    present_employment_since: Literal["A71", "A72", "A73", "A74", "A75"]
    installment_rate_in_percentage_of_disposable_income: int
    personal_status_and_sex: Literal["A91", "A92", "A93", "A94"]
    other_debtors_guarantors: Literal["A101", "A102", "A103"]
    present_residence_since: int
    property: Literal["A121", "A122", "A123", "A124"]
    age_in_years: int = Field(validation_alias=AliasChoices("age_in_years", "age"))
    other_installment_plans: Literal["A141", "A142", "A143"]
    housing: Literal["A151", "A152", "A153"]
    number_of_existing_credits_at_this_bank: int
    job: Literal["A171", "A172", "A173", "A174"]
    number_of_people_being_liable_to_provide_maintenance_for: int
    telephone: Literal["A191", "A192"]
    foreign_worker: Literal["A201", "A202"]


class PredictResponse(BaseModel):
    pd: float
    decision: str
    review_required: bool
    decision_reason: str
    applicant_expected_loss: float
    policy_version: str
    policy_low_threshold: float
    policy_high_threshold: float


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [column.lower().replace(" ", "_") for column in result.columns]
    return result


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Local dataset not found at {DATA_PATH}. "
            "Add the German Credit dataset file before starting the API."
        )

    df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=DATA_COLUMNS)
    df = normalize_columns(df)
    df["target"] = df["target"].map({1: 0, 2: 1})
    return df


def make_decision(pd_value: float, low: float, high: float) -> str:
    if pd_value < low:
        return "Approve"
    if pd_value < high:
        return "Review"
    return "Reject"


def prettify_feature(feature: str) -> str:
    return FEATURE_LABEL_MAP.get(feature, feature)


def generate_reason(contributions: list[tuple[str, float]]) -> str:
    reasons = []
    for feature, value in contributions[:3]:
        label = prettify_feature(feature)
        direction = "increases risk" if value > 0 else "decreases risk"
        reasons.append(f"{label} {direction}")
    return "; ".join(reasons)


def build_processed_frame(preprocessor: ColumnTransformer, raw_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    processed = preprocessor.transform(raw_df)
    return pd.DataFrame(processed, columns=feature_names)


def select_policy(pd_scores: pd.Series, y_test: pd.Series, credit_amount: pd.Series) -> dict[str, Any]:
    policy_rows = []

    for low, high in THRESHOLD_SETS:
        decisions = [make_decision(score, low, high) for score in pd_scores]
        temp_df = pd.DataFrame(
            {
                "actual": y_test.values,
                "pd": pd_scores,
                "decision": decisions,
                "loan_amount": credit_amount.values,
            }
        )
        temp_df["el"] = temp_df["pd"] * LGD * temp_df["loan_amount"]

        approval_rate = (temp_df["decision"] == "Approve").mean()
        missed_defaults = temp_df[
            (temp_df["actual"] == 1) & (temp_df["decision"] == "Approve")
        ].shape[0]
        false_rejects = temp_df[
            (temp_df["actual"] == 0) & (temp_df["decision"] == "Reject")
        ].shape[0]
        approve_el = temp_df.loc[temp_df["decision"] == "Approve", "el"].sum()
        review_el = temp_df.loc[temp_df["decision"] == "Review", "el"].sum()
        booked_expected_loss = approve_el + REVIEW_BOOKING_RATE * review_el

        policy_rows.append(
            {
                "low": low,
                "high": high,
                "thresholds": f"{low}-{high}",
                "approval_rate": approval_rate,
                "missed_defaults": missed_defaults,
                "false_rejects": false_rejects,
                "booked_expected_loss": booked_expected_loss,
            }
        )

    policy_df = pd.DataFrame(policy_rows)
    candidate_policies = policy_df[
        (policy_df["approval_rate"] >= MIN_APPROVAL_RATE)
        & (policy_df["missed_defaults"] <= MAX_MISSED_DEFAULTS)
    ]

    if not candidate_policies.empty:
        best_policy = candidate_policies.sort_values(
            by=["booked_expected_loss", "false_rejects"]
        ).iloc[0]
    else:
        best_policy = policy_df.sort_values(
            by=["booked_expected_loss", "missed_defaults"]
        ).iloc[0]

    return best_policy.to_dict()


def train_artifacts() -> dict[str, Any]:
    df = load_dataset()
    x = df.drop(columns=["target"])
    y = df["target"]

    categorical_cols = x.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = x.select_dtypes(exclude=["object"]).columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            )
        ],
        remainder="passthrough",
    )

    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)

    encoded_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
    all_feature_names = list(encoded_feature_names) + numerical_cols

    x_train_df = pd.DataFrame(x_train_processed, columns=all_feature_names)
    x_test_df = pd.DataFrame(x_test_processed, columns=all_feature_names)

    model = LogisticRegression(max_iter=3000, solver="liblinear")
    model.fit(x_train_df, y_train)

    test_pd_scores = pd.Series(model.predict_proba(x_test_df)[:, 1])
    best_policy = select_policy(test_pd_scores, y_test.reset_index(drop=True), x_test.reset_index(drop=True)["credit_amount"])

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": all_feature_names,
        "best_policy": best_policy,
        "background_df": x_train_df,
    }


def save_artifacts(artifacts: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "model": artifacts["model"],
            "preprocessor": artifacts["preprocessor"],
            "feature_names": artifacts["feature_names"],
            "best_policy": artifacts["best_policy"],
            "background_df": artifacts["background_df"],
        },
        ARTIFACTS_PATH,
    )


def load_or_train_artifacts() -> dict[str, Any]:
    if ARTIFACTS_PATH.exists():
        artifacts = load(ARTIFACTS_PATH)
    else:
        artifacts = train_artifacts()
        save_artifacts(artifacts)

    artifacts["explainer"] = shap.Explainer(artifacts["model"], artifacts["background_df"])
    return artifacts


@asynccontextmanager
async def lifespan(_: FastAPI):
    app.state.artifacts = load_or_train_artifacts()
    yield


app = FastAPI(
    title="Credit Risk Decision API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    artifacts = app.state.artifacts
    return {
        "status": "ok",
        "model": "logistic_regression",
        "policy_version": artifacts["best_policy"]["thresholds"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(input_data: BorrowerInput) -> PredictResponse:
    artifacts = app.state.artifacts
    raw_payload = input_data.model_dump()

    raw_df = pd.DataFrame(
        [
            {
                raw_column: raw_payload[api_field]
                for raw_column, api_field in RAW_TO_API_FIELD.items()
            }
        ]
    )

    processed_df = build_processed_frame(
        artifacts["preprocessor"],
        raw_df,
        artifacts["feature_names"],
    )

    pd_value = float(artifacts["model"].predict_proba(processed_df)[0, 1])
    low = float(artifacts["best_policy"]["low"])
    high = float(artifacts["best_policy"]["high"])
    decision = make_decision(pd_value, low, high)

    shap_values = artifacts["explainer"](processed_df)
    contributions = list(zip(processed_df.columns, shap_values.values[0]))
    contributions_sorted = sorted(contributions, key=lambda item: abs(item[1]), reverse=True)
    reason = generate_reason(contributions_sorted)

    return PredictResponse(
        pd=pd_value,
        decision=decision,
        review_required=decision == "Review",
        decision_reason=reason,
        applicant_expected_loss=float(pd_value * LGD * raw_payload["credit_amount"]),
        policy_version=str(artifacts["best_policy"]["thresholds"]),
        policy_low_threshold=low,
        policy_high_threshold=high,
    )
