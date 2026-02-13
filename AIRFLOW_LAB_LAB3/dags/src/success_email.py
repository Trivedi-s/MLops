from airflow.hooks.base import BaseHook
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_success_email(**kwargs):
    print("=" * 50)
    print("TASK 5: Sending Success Email")
    print("=" * 50)
    
    conn = BaseHook.get_connection('smtp_default')
    sender_email = conn.login
    password = conn.password
    receiver_email = 'trivedishloka13@gmail.com'
    
    subject = 'Airflow Success: ML_Pipeline - All Tasks Completed'
    body = '''Hi Shloka,

The ML Pipeline DAG completed successfully!

Pipeline Tasks:
- load_data_task: SUCCESS
- data_preprocessing_task: SUCCESS  
- build_model_task: SUCCESS
- evaluate_model_task: SUCCESS
- send_email_task: SUCCESS

Model: Random Forest Classifier
Output: random_forest_model.sav
Metrics: model_metrics.txt

Best regards,
Airflow
'''
    
    email_message = MIMEMultipart()
    email_message['Subject'] = subject
    email_message['From'] = sender_email
    email_message['To'] = receiver_email
    email_message.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, email_message.as_string())
        print(f"Success email sent to {receiver_email}!")
        server.quit()
    except Exception as e:
        print(f"Error sending email: {e}")
        raise e
    
    print("=" * 50)