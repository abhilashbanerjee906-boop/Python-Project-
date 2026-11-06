"""
STUDENT PERFORMANCE AND STUDY-HOURS DASHBOARD
One-file Python project.

Features:
- Anonymised synthetic student records
- Score/grade functions and StudentRecord class
- Dictionaries and CSV file handling
- NumPy averages, median, and standard deviation
- Pandas course-level summaries
- Matplotlib scatter plot, score histogram, and course average chart
- IQR-based score outlier review and correlation comparison

The data is synthetic and created for demonstration; it is not real student data.
"""

from pathlib import Path
import csv
import numpy as np
import pandas as pd
import matplotlib

# Use a non-interactive backend so the script also works from Windows Command Prompt.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# All files are saved beside this Python script.
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "students.csv"
OUTPUT_DIR = BASE_DIR / "output"
GRAPH_DIR = BASE_DIR / "graphs"


def calculate_average(values):
    """Return the mean of a sequence; return 0 for an empty sequence."""
    return float(sum(values) / len(values)) if len(values) else 0.0


def calculate_grade(score):
    """Convert a score out of 100 into a letter grade."""
    score = float(score)
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    return "F"


def validate_record(study_hours, score):
    """Raise a clear error for invalid study hours or assessment scores."""
    if float(study_hours) < 0:
        raise ValueError("Study hours cannot be negative.")
    if not 0 <= float(score) <= 100:
        raise ValueError("Score must be between 0 and 100.")


class StudentRecord:
    """Represent one anonymised student record."""

    def __init__(self, student_id, course, study_hours, score):
        validate_record(study_hours, score)
        self.student_id = str(student_id)
        self.course = str(course)
        self.study_hours = float(study_hours)
        self.score = float(score)
        self.grade = calculate_grade(self.score)

    def to_dict(self):
        return {
            "student_id": self.student_id,
            "course": self.course,
            "study_hours": self.study_hours,
            "score": self.score,
            "grade": self.grade,
        }


def create_sample_records():
    """Create 40 fictional, anonymised records with a mild positive association."""
    courses = ["Python", "Mathematics", "Physics", "Electronics"]
    hours = [
        1.0, 2.5, 3.0, 4.0, 5.5, 2.0, 6.0, 7.0, 3.5, 8.0,
        4.5, 1.5, 6.5, 5.0, 9.0, 2.8, 7.5, 3.2, 4.8, 6.2,
        1.8, 5.8, 8.5, 2.2, 7.2, 4.2, 3.8, 6.8, 5.2, 9.5,
        2.6, 4.6, 7.8, 1.2, 8.2, 5.6, 3.4, 6.4, 4.4, 7.4
    ]
    noise = [
        4, -8, 3, 5, -6, 9, -2, 1, -4, 2,
        7, -3, 0, 4, -5, 6, -7, 2, 1, -1,
        5, -4, 3, -6, 0, 8, -2, 4, -3, 1,
        -5, 6, -1, 3, -4, 2, 5, -2, 7, -3
    ]

    records = []
    for index, (study_hours, variation) in enumerate(zip(hours, noise), start=1):
        course = courses[(index - 1) % len(courses)]
        score = max(0, min(100, round(43 + 5.4 * study_hours + variation, 1)))

        # One unusual synthetic score is included to demonstrate outlier review.
        if index == 37:
            score = 18.0

        record = StudentRecord(
            student_id=f"S{index:03d}",
            course=course,
            study_hours=study_hours,
            score=score
        )
        records.append(record.to_dict())
    return records


def save_and_load_csv(records):
    """Write anonymised records to CSV, then load them back as dictionaries."""
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["student_id", "course", "study_hours", "score", "grade"]

    with open(DATA_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    loaded_records = []
    with open(DATA_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            validate_record(row["study_hours"], row["score"])
            # Dictionaries are used to store each record.
            loaded_records.append({
                "student_id": row["student_id"],
                "course": row["course"],
                "study_hours": float(row["study_hours"]),
                "score": float(row["score"]),
                "grade": calculate_grade(row["score"]),
            })
    return loaded_records


def analyse_records(records):
    """Calculate NumPy statistics and Pandas course summaries."""
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No student records are available.")

    scores = df["score"].to_numpy(dtype=float)
    study_hours = df["study_hours"].to_numpy(dtype=float)

    print("\n" + "=" * 58)
    print("STUDENT PERFORMANCE AND STUDY-HOURS DASHBOARD")
    print("=" * 58)
    print(f"Number of anonymised records : {len(df)}")
    print(f"Mean assessment score        : {np.mean(scores):.2f}")
    print(f"Median assessment score      : {np.median(scores):.2f}")
    print(f"Score standard deviation     : {np.std(scores):.2f}")
    print(f"Mean reported study hours    : {np.mean(study_hours):.2f}")
    print(f"Mean score (function)        : {calculate_average(scores):.2f}")

    course_summary = df.groupby("course").agg(
        students=("student_id", "count"),
        average_score=("score", "mean"),
        median_score=("score", "median"),
        average_study_hours=("study_hours", "mean"),
        score_std=("score", "std")
    ).round(2)

    print("\nCOURSE SUMMARY (Pandas)")
    print(course_summary.to_string())

    correlation_all = df["study_hours"].corr(df["score"])
    print("\nSTUDY HOURS AND SCORES")
    print(f"Pearson correlation, all records: {correlation_all:.3f}")

    # IQR outlier rule: flag unusual scores for review, not automatic deletion.
    q1 = df["score"].quantile(0.25)
    q3 = df["score"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_mask = (df["score"] < lower_bound) | (df["score"] > upper_bound)
    outliers = df.loc[outlier_mask, ["student_id", "course", "study_hours", "score"]]

    print("\nOUTLIER CHECK (IQR rule on assessment scores)")
    print(f"Review bounds: {lower_bound:.2f} to {upper_bound:.2f}")
    if outliers.empty:
        print("No scores were flagged by the IQR rule.")
        filtered_df = df.copy()
    else:
        print("Flagged record(s) — review rather than deleting automatically:")
        print(outliers.to_string(index=False))
        filtered_df = df.loc[~outlier_mask].copy()

    if (
        len(filtered_df) >= 2
        and filtered_df["study_hours"].nunique() > 1
        and filtered_df["score"].nunique() > 1
    ):
        correlation_filtered = filtered_df["study_hours"].corr(filtered_df["score"])
        print(f"Pearson correlation after excluding flagged score(s): {correlation_filtered:.3f}")
        print("Compare the correlations to judge sensitivity to the flagged record(s).")
    else:
        print("Not enough remaining variation to calculate the filtered correlation.")

    print("\nINTERPRETATION")
    print("Correlation describes association; it does not prove causation.")
    print("Other factors such as prior knowledge, sleep, assessment difficulty,")
    print("and reporting accuracy may also influence assessment scores.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "student_records_with_grades.csv", index=False)
    course_summary.to_csv(OUTPUT_DIR / "course_summary.csv")
    return df, course_summary


def create_graphs(df, course_summary):
    """Save three graphs as PNG images in the graphs folder."""
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    # Graph 1: scatter plot and descriptive trend line.
    plt.figure(figsize=(8, 5))
    plt.scatter(df["study_hours"], df["score"], alpha=0.8)
    if df["study_hours"].nunique() > 1:
        slope, intercept = np.polyfit(df["study_hours"], df["score"], 1)
        x_values = np.linspace(df["study_hours"].min(), df["study_hours"].max(), 100)
        plt.plot(x_values, slope * x_values + intercept, linestyle="--",
                 label="Linear trend (descriptive)")
        plt.legend()
    plt.title("Study Hours vs Assessment Score")
    plt.xlabel("Reported study hours")
    plt.ylabel("Assessment score (out of 100)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "study_hours_vs_score.png", dpi=160)
    plt.close()

    # Graph 2: score histogram.
    plt.figure(figsize=(8, 5))
    plt.hist(df["score"], bins=10, edgecolor="black")
    plt.title("Distribution of Assessment Scores")
    plt.xlabel("Assessment score (out of 100)")
    plt.ylabel("Number of students")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "score_histogram.png", dpi=160)
    plt.close()

    # Graph 3: average score by course.
    means = course_summary["average_score"].sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    plt.bar(means.index, means.values)
    plt.title("Average Assessment Score by Course")
    plt.xlabel("Course")
    plt.ylabel("Average score (out of 100)")
    plt.ylim(0, 100)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "average_score_by_course.png", dpi=160)
    plt.close()

    print("\nGRAPHS SAVED")
    print(f"1. {GRAPH_DIR / 'study_hours_vs_score.png'}")
    print(f"2. {GRAPH_DIR / 'score_histogram.png'}")
    print(f"3. {GRAPH_DIR / 'average_score_by_course.png'}")


def main():
    try:
        records = create_sample_records()
        loaded_records = save_and_load_csv(records)
        df, course_summary = analyse_records(loaded_records)
        create_graphs(df, course_summary)

        print("\nPROJECT COMPLETED SUCCESSFULLY.")
        print(f"CSV data: {DATA_FILE}")
        print(f"Output summaries: {OUTPUT_DIR}")
        print(f"Graphs: {GRAPH_DIR}")
    except Exception as error:
        print("\nThe project could not finish.")
        print("Error:", error)
        print("Check that NumPy, Pandas, and Matplotlib are installed.")
        raise


if __name__ == "__main__":
    main()
