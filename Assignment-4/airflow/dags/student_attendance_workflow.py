from airflow.sdk import dag, task
from datetime import datetime


@dag(
    dag_id="student_attendance_workflow",
    start_date=datetime(2026, 8, 1),
    schedule=None,
    catchup=False
)
def student_attendance_workflow():

    @task
    def collect_data():
        print("Step 1: Collecting student data...")

        total_students = 100

        print(f"Total Students: {total_students}")

        return total_students

    @task
    def calculate_attendance(total_students):
        print("Step 2: Calculating attendance...")

        present_students = 85

        attendance = (present_students / total_students) * 100

        print(f"Present Students: {present_students}")
        print(f"Attendance: {attendance}%")

        return attendance

    @task
    def generate_report(attendance):
        print("Step 3: Generating report...")
        print("---------------------------")
        print(" STUDENT ATTENDANCE REPORT")
        print("---------------------------")
        print(f"Attendance: {attendance}%")

        if attendance >= 75:
            print("Status: GOOD")
        else:
            print("Status: LOW")

        print("Report Generated Successfully!")


    total_students = collect_data()
    attendance = calculate_attendance(total_students)
    generate_report(attendance)


student_attendance_workflow()