from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "german.data"
OUTPUT_PATH = BASE_DIR / "data" / "german_decoded.csv"

COLUMNS = [
    "status_of_existing_checking_account",
    "duration_in_month",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account_bonds",
    "present_employment_since",
    "installment_rate_in_percentage_of_disposable_income",
    "personal_status_and_sex",
    "other_debtors_guarantors",
    "present_residence_since",
    "property",
    "age_in_years",
    "other_installment_plans",
    "housing",
    "number_of_existing_credits_at_this_bank",
    "job",
    "number_of_people_being_liable_to_provide_maintenance_for",
    "telephone",
    "foreign_worker",
    "target",
]

VALUE_MAPS = {
    "status_of_existing_checking_account": {
        "A11": "checking account below 0 DM",
        "A12": "checking account between 0 and 200 DM",
        "A13": "checking account above 200 DM or salary assignment for at least 1 year",
        "A14": "no checking account",
    },
    "credit_history": {
        "A30": "no credits taken or all credits paid back duly",
        "A31": "all credits at this bank paid back duly",
        "A32": "existing credits paid back duly till now",
        "A33": "delay in paying off in the past",
        "A34": "critical account or other credits existing",
    },
    "purpose": {
        "A40": "car (new)",
        "A41": "car (used)",
        "A42": "furniture or equipment",
        "A43": "radio or television",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "others",
    },
    "savings_account_bonds": {
        "A61": "savings below 100 DM",
        "A62": "savings between 100 and 500 DM",
        "A63": "savings between 500 and 1000 DM",
        "A64": "savings above 1000 DM",
        "A65": "unknown or no savings account",
    },
    "present_employment_since": {
        "A71": "unemployed",
        "A72": "employed less than 1 year",
        "A73": "employed between 1 and 4 years",
        "A74": "employed between 4 and 7 years",
        "A75": "employed 7 years or more",
    },
    "personal_status_and_sex": {
        "A91": "male divorced or separated",
        "A92": "female divorced, separated, or married",
        "A93": "male single",
        "A94": "male married or widowed",
        "A95": "female single",
    },
    "other_debtors_guarantors": {
        "A101": "none",
        "A102": "co-applicant",
        "A103": "guarantor",
    },
    "property": {
        "A121": "real estate",
        "A122": "building society savings agreement or life insurance",
        "A123": "car or other property",
        "A124": "unknown or no property",
    },
    "other_installment_plans": {
        "A141": "bank",
        "A142": "stores",
        "A143": "none",
    },
    "housing": {
        "A151": "rent",
        "A152": "own",
        "A153": "for free",
    },
    "job": {
        "A171": "unemployed or unskilled non-resident",
        "A172": "unskilled resident",
        "A173": "skilled employee or official",
        "A174": "management, self-employed, or highly qualified",
    },
    "telephone": {
        "A191": "no telephone",
        "A192": "telephone registered under customer name",
    },
    "foreign_worker": {
        "A201": "yes",
        "A202": "no",
    },
    "target": {
        1: "good credit",
        2: "bad credit",
    },
}


def main() -> None:
    df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=COLUMNS)
    decoded = df.copy()

    for column, mapping in VALUE_MAPS.items():
        decoded[column] = decoded[column].map(mapping).fillna(decoded[column])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    decoded.to_csv(OUTPUT_PATH, index=False)
    print(f"Decoded dataset written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
