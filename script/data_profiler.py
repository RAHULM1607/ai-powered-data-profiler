import os
import sys
import json
import shutil
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Base project folder: can be passed as argument (e.g. sys.argv[1]), otherwise defaults to D:\python_deai
PROJECT_FOLDER = Path(sys.argv[1])
LANDING = PROJECT_FOLDER / "landing"
DATA = PROJECT_FOLDER / "data"

PROCESSING = DATA / "processing"
ARCHIVE = DATA / "archive"
ERROR = DATA / "error"
OUTPUT = DATA / "output"

for folder in [LANDING, PROCESSING, ARCHIVE, ERROR, OUTPUT]:
    folder.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_FOLDER / ".env")

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

MODEL = "gemini-3.5-flash-lite"


def load_dataframe(file_path):
    extension = file_path.suffix.lower()

    if extension == ".csv":
        return pd.read_csv(file_path)
    if extension == ".json":
        return pd.read_json(file_path)
    if extension == ".xlsx":
        return pd.read_excel(file_path)
    if extension == ".txt":
        return pd.read_csv(file_path, sep="|")

    print(f"Unsupported file type: {extension}")
    return None


def profile_dataframe(df, file_name):
    print("\nFirst 5 records:")
    print(df.head())
    print("\nShape:")
    print(df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print("\nDataFrame information:")
    df.info()

    print("\nStatistical information:")
    print(df.describe(include="all"))
    print("\nNull values:")
    print(df.isnull().sum())
    print("\nUnique values:")
    print(df.nunique())
    duplicates = df.duplicated().sum()
    print("\nDuplicate records:", duplicates)

    row_count = len(df)
    if row_count:
        null_percentage = (df.isnull().sum() / row_count * 100).round(2)
    else:
        null_percentage = df.isnull().sum()

    column_profile = pd.DataFrame({
        "column_name": df.columns,
        "data_type": df.dtypes.astype(str).values,
        "null_count": df.isnull().sum().values,
        "actual_count": row_count,
        "unique_count": df.nunique().values,
        "null_percentage": null_percentage.values
    })

    profile_file = OUTPUT / f"{Path(file_name).stem}_profile.csv"
    column_profile.to_csv(profile_file, index=False)
    print("\nProfile saved:", profile_file)

    data_types = df.dtypes.astype(str).to_dict()
    return data_types, column_profile, duplicates

def run_data_quality_checks(df, file_name):
    """Check common data-quality issues and save a JSON report."""

    issues = []

    # 1. Missing values
    for column in df.columns:
        missing_count = int(df[column].isna().sum())

        if missing_count > 0:
            issues.append({
                "column": column,
                "issue": "Missing values",
                "count": missing_count
            })

    # 2. Duplicate rows
    duplicate_count = int(df.duplicated().sum())

    if duplicate_count > 0:
        issues.append({
            "column": None,
            "issue": "Duplicate rows",
            "count": duplicate_count
        })

    # 3. Duplicate values in ID columns
    for column in df.columns:
        if "id" in column.lower():
            duplicate_count = int(
                df[column].dropna().duplicated().sum()
            )

            if duplicate_count > 0:
                issues.append({
                    "column": column,
                    "issue": "Duplicate ID values",
                    "count": duplicate_count
                })

    # 4. Invalid email addresses
    for column in df.columns:
        if "email" in column.lower():
            values = df[column].dropna().astype(str)
            valid_email = values.str.match(
                r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
            )
            invalid_count = int((~valid_email).sum())

            if invalid_count > 0:
                issues.append({
                    "column": column,
                    "issue": "Invalid email addresses",
                    "count": invalid_count
                })

    # 5. Negative values in selected numeric columns
    for column in df.select_dtypes(include="number").columns:
        if any(word in column.lower()
               for word in ["age", "salary", "amount", "price"]):

            negative_count = int((df[column] < 0).sum())

            if negative_count > 0:
                issues.append({
                    "column": column,
                    "issue": "Negative values",
                    "count": negative_count
                })

    report = {
        "file_name": file_name,
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "total_issues_found": len(issues),
        "issues": issues
    }

    report_path = OUTPUT / (
        f"{Path(file_name).stem}_quality_report.json"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nData quality report saved:", report_path)
    print("Quality issues found:", len(issues))

    return report


def classify_data_locally(df, file_name):
    """Classify dataset columns using simple name-based rules."""

    column_classifications = []

    for column in df.columns:
        name = column.lower()

        if "email" in name:
            category = "Email Address"
            pii = True
        elif "phone" in name or "mobile" in name:
            category = "Phone Number"
            pii = True
        elif "address" in name:
            category = "Physical Address"
            pii = True
        elif "dateofbirth" in name or "birth_date" in name or name == "dob":
            category = "Date of Birth"
            pii = True
        elif "first_name" in name or "firstname" in name:
            category = "Personal Name"
            pii = True
        elif "last_name" in name or "lastname" in name:
            category = "Personal Name"
            pii = True
        elif "id" in name:
            category = "Identifier"
            pii = False
        elif "date" in name or "hiredate" in name:
            category = "Date"
            pii = False
        else:
            category = "General Data"
            pii = False

        column_classifications.append({
            "column": column,
            "category": category,
            "possible_pii": pii,
            "review_required": pii
        })

    result = {
        "file_name": file_name,
        "classification_method": "Local rule-based classification",
        "dataset_columns": column_classifications
    }

    output_file = OUTPUT / (
        f"{Path(file_name).stem}_classification.json"
    )

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print("\nLocal classification saved:", output_file)

    return result


def create_data_layers(df, file_name):
    from pathlib import Path

    base_folder = PROJECT_FOLDER / "data"

    bronze_folder = base_folder / "bronze"
    silver_folder = base_folder / "silver"
    gold_folder = base_folder / "gold"

    # Create folders if they do not exist
    for folder in [bronze_folder, silver_folder, gold_folder]:
        folder.mkdir(parents=True, exist_ok=True)

    file_stem = Path(file_name).stem

    # BRONZE: Save the original loaded data without cleaning
    bronze_path = bronze_folder / f"{file_stem}.csv"
    df.to_csv(bronze_path, index=False)

    # SILVER: Clean column names, text spaces, and duplicate rows

    # SILVER: Clean and validate data
    silver_df = df.copy()

    # 1. Remove extra spaces from column names
    silver_df.columns = silver_df.columns.str.strip()

    # 2. Remove extra spaces from text values
    for column in silver_df.select_dtypes(include=["object", "str"]).columns:
        silver_df[column] = silver_df[column].map(
            lambda value: value.strip()
            if isinstance(value, str) else value
        )

    # 3. Convert empty text values to missing values
    silver_df = silver_df.replace(r"^\s*$", pd.NA, regex=True)

    # 4. Remove duplicate rows
    silver_df = silver_df.drop_duplicates()

    # 5. Convert known date columns to proper dates
    for column in silver_df.columns:
        if column.lower() in ["hiredate", "dateofbirth", "birth_date", "dob"]:
            original_dates = silver_df[column].copy()

            converted = pd.to_datetime(
                original_dates,
                format="mixed",
                dayfirst=True,
                errors="coerce"
            )



            # Only convert if all existing non-null dates are valid
            original_non_null = silver_df[column].notna()
            if converted[original_non_null].notna().all():
                silver_df[column] = converted.dt.strftime("%Y-%m-%d")

    # 6. Remove records with missing primary identifiers
    for column in ["EmployeeID", "CustomerID"]:
        if column in silver_df.columns:
            missing_ids = silver_df[column].isna().sum()

            if missing_ids > 0:
                print(f"Silver: Excluding {missing_ids} rows with missing {column}")

                silver_df = silver_df.dropna(subset=[column])

    # 7. Save cleaned data
    silver_path = silver_folder / f"{file_stem}.csv"
    silver_df.to_csv(silver_path, index=False)

    # GOLD: Create a summary for reporting
    summary = pd.DataFrame({
        "Metric": [
            "Total Rows",
            "Total Columns",
            "Duplicate Rows",
            "Missing Values"
        ],
        "Value": [
            len(silver_df),
            len(silver_df.columns),
            int(silver_df.duplicated().sum()),
            int(silver_df.isnull().sum().sum())
        ]
    })

    gold_path = gold_folder / f"{file_stem}_summary.csv"
    summary.to_csv(gold_path, index=False)

    print(f"Bronze saved: {bronze_path}")
    print(f"Silver saved: {silver_path}")
    print(f"Gold saved: {gold_path}")

def generate_schema_locally(df, file_name):
    from pathlib import Path
    import json

    stem = Path(file_name).stem
    schema = {}

    for column in df.columns:
        dtype = str(df[column].dtype)

        if "int" in dtype:
            sql_type = "INTEGER"
        elif "float" in dtype:
            sql_type = "FLOAT"
        elif "datetime" in dtype:
            sql_type = "DATE"
        else:
            sql_type = "VARCHAR(255)"

        schema[column] = {
            "pandas_type": dtype,
            "sql_type": sql_type,
            "nullable": bool(df[column].isna().any())
        }

    schema_path = OUTPUT / f"{stem}_schema.json"
    schema_path.write_text(
        json.dumps(schema, indent=4),
        encoding="utf-8"
    )

    print("Local schema saved:", schema_path)
    return schema


def build_prompts(file_name, df, data_types, column_profile):
    sample_json = df.head(10).astype(str).to_json(
        orient="records",
        indent=2,
        force_ascii=False
    )

    columns = df.columns.tolist()
    types_json = json.dumps(data_types, indent=2, ensure_ascii=False)
    profile_text = column_profile.to_string(index=False)

    classification_prompt = f"""
You are a Data Engineering and Data Governance expert.

Analyze this dataset.

File: {file_name}
Columns: {json.dumps(columns)}
Pandas data types:
{types_json}

Column profile:
{profile_text}

Sample records:
{sample_json}

Identify:
1. Business domain
2. Dataset type
3. PII columns
4. Sensitive columns
5. Identifier columns
6. Business measure columns
7. Date columns
8. Categorical columns

Explain the classification of every column.
Return valid JSON only, without Markdown fences.
Use only columns that exist in the dataset.
If a classification cannot be determined, state "unknown".
Do not infer sensitive personal attributes from names or weak evidence.
Flag possible PII for human review.
"""

    schema_prompt = f"""
You are a Senior Data Engineer.

Analyze this source dataset.

File: {file_name}
Columns: {json.dumps(columns)}
Pandas data types:
{types_json}

Column profile:
{profile_text}

Sample records:
{sample_json}

Determine:
1. Dataset business purpose
2. Recommended target table
3. Candidate primary key
4. Recommended partition column, if appropriate
5. Business meaning of every column
6. Recommended target data type
7. Nullable or not, based on evidence
8. Possible foreign keys
9. Data quality concerns

Return valid JSON only, without Markdown fences.
Use only columns that actually exist.
Do not claim a primary key or foreign key is confirmed unless evidence establishes it.
Label uncertain recommendations clearly.
"""
    return classification_prompt, schema_prompt


def ask_gemini(prompt):
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt)
    return response.text



def process_file(source_file):
    file_name = source_file.name
    processing_file = PROCESSING / file_name

    print("\nProcessing:", file_name)
    print("File size:", source_file.stat().st_size, "bytes")

    try:
        shutil.move(str(source_file), str(processing_file))

        df = load_dataframe(processing_file)
        create_data_layers(df, file_name)

        if df is None or len(df.columns) == 0:
            raise ValueError("Unsupported file or dataset has no columns.")

        data_types, column_profile, duplicates = profile_dataframe(
            df, file_name
        )
        quality_report = run_data_quality_checks(df, file_name)
        classification_result = classify_data_locally(df, file_name)

        # Save a CSV copy even if Gemini is unavailable.
        processed_file = OUTPUT / f"{Path(file_name).stem}_processed.csv"
        df.to_csv(processed_file, index=False)

        classification_prompt, schema_prompt = build_prompts(
            file_name, df, data_types, column_profile
        )

        try:
            print("\nRequesting AI classification with", MODEL, "...")
            classification = ask_gemini(classification_prompt)

            print("\nRequesting target schema recommendations with", MODEL, "...")
            schema = ask_gemini(schema_prompt)

            classification_file = (
                OUTPUT / f"{Path(file_name).stem}_classification.json"
            )
            schema_file = OUTPUT / f"{Path(file_name).stem}_schema.json"

            classification_file.write_text(
                classification, encoding="utf-8"
            )
            schema_file.write_text(schema, encoding="utf-8")



        except Exception as ai_error:

            # Record the AI failure, but continue local processing.

            error_file = OUTPUT / f"{Path(file_name).stem}_ai_error.txt"

            error_file.write_text(

                f"File: {file_name}\n"

                f"AI classification/schema generation failed:\n"

                f"{type(ai_error).__name__}: {ai_error}\n",

                encoding="utf-8"

            )

            print("\nAI processing failed:", ai_error)

            print("Continuing with local processing.")

            print("Local classification:", OUTPUT / f"{Path(file_name).stem}_classification.json")

            print("AI error report:", error_file)

            # Generate a local schema if Gemini is unavailable.

            generate_schema_locally(df, file_name)

               # Archive the source after local processing, even if AI failed.
        shutil.move(str(processing_file), str(ARCHIVE / file_name))

        print("\nSUCCESS:", file_name)
        print("Profile:", OUTPUT / f"{Path(file_name).stem}_profile.csv")
        print("Classification:", classification_file)
        print("Schema:", schema_file)
        print("Processed CSV:", processed_file)
        print("Duplicate records detected:", duplicates)
        print("Source archived:", ARCHIVE / file_name)

    except Exception as error:
        print(f"\nProcessing failed for {file_name}: {error}")

        # Move a remaining processing file to error, if possible.
        if processing_file.exists():
            error_destination = ERROR / file_name
            if not error_destination.exists():
                shutil.move(str(processing_file), str(error_destination))
                print("Moved failed file to:", error_destination)
            else:
                print(
                    "Error destination already exists. "
                    "File remains in processing:",
                    processing_file
                )


def main():
    supported_extensions = [".csv", ".json", ".xlsx", ".txt"]
    files = []

    for f in LANDING.iterdir():
        if f.is_file():
            if f.suffix.lower() in supported_extensions:
                files.append(f)
            else:
                print("\nUnsupported file:", f.name)
                shutil.move(str(f), str(ERROR / f.name))
                print("Moved to:", ERROR / f.name)

    files.sort()

    print("Data Engineering Data Profiler")
    print("Gemini model:", MODEL)

    print("\nFiles available in landing:")
    for file_path in files:
        print(file_path.name)

    if not files:
        print("\nNo files found in landing.")
        print("Place CSV, JSON, XLSX, or TXT files there.")
        return

    for source_file in files:
        process_file(source_file)

    print("Pipeline execution completed.")
    print("Check data\\output, data\\archive, and data\\error.")


main()