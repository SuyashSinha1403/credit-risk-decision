# German Credit Data Dictionary

This project includes the original raw dataset in [german.data](D:/n8n/data/german.data) and a decoded human-readable version in `german_decoded.csv`.

## Key idea

The original dataset stores many categorical values as compact codes such as:

- `A11`
- `A32`
- `A143`

These are the original UCI dataset category values.  
The decoded CSV translates them into plain language for easier understanding.

## Main fields

- `status_of_existing_checking_account`: checking account status
- `duration_in_month`: loan duration
- `credit_history`: past repayment history
- `purpose`: purpose of the loan
- `credit_amount`: requested loan amount
- `savings_account_bonds`: savings level
- `present_employment_since`: employment duration
- `installment_rate_in_percentage_of_disposable_income`: installment burden
- `personal_status_and_sex`: marital status and sex category from the original dataset
- `other_debtors_guarantors`: co-applicant or guarantor information
- `present_residence_since`: years at current residence
- `property`: property category
- `age_in_years`: borrower age
- `other_installment_plans`: other installment plans
- `housing`: rent / own / free
- `number_of_existing_credits_at_this_bank`: number of current credits
- `job`: employment type
- `number_of_people_being_liable_to_provide_maintenance_for`: maintenance obligations
- `telephone`: telephone availability
- `foreign_worker`: foreign worker flag
- `target`: credit outcome label

## Target meaning

- `good credit`: non-default / safer case
- `bad credit`: default / riskier case

## Example code translations

- `A11` -> checking account below 0 DM
- `A12` -> checking account between 0 and 200 DM
- `A14` -> no checking account
- `A32` -> existing credits paid back duly till now
- `A33` -> delay in paying off in the past
- `A43` -> radio or television
- `A61` -> savings below 100 DM
- `A73` -> employed between 1 and 4 years
- `A101` -> none
- `A121` -> real estate
- `A143` -> none
- `A152` -> own
- `A173` -> skilled employee or official
- `A201` -> yes

## Why both files are included

- `german.data` keeps the original raw UCI format for reproducibility
- `german_decoded.csv` makes the dataset understandable for reviewers and GitHub readers
