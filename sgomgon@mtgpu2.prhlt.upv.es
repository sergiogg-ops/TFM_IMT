#!/usr/bin/env python3
import argparse
import subprocess
import sys
import smtplib
from email.mime.text import MIMEText

def send_email(subject, body):
    # Configure your SMTP settings here.
    smtp_server   = 'smtp.gmail.com'
    smtp_port     = 587
    smtp_user     = 'entrenadorbot3@gmail.com'
    smtp_password = 'xhpffxbidpkunfyw '
    sender_email  = 'entrenadorbot3@gmail.com'
    receiver_email= 'sergiogg1yo@gmail. com'

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
    except Exception as e:
        print(f"Failed to send email: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Envelope script to call other Python routines with provided arguments.'
    )
    parser.add_argument(
        'script',
        help='Path to the Python script to execute (e.g., subroutine.py)'
    )
    parser.add_argument(
        'args',
        nargs=argparse.REMAINDER,
        help='Arguments to pass to the subroutine'
    )
    args = parser.parse_args()

    # Assemble the command to call the subroutine using the same Python interpreter.
    cmd = [sys.executable, args.script] + args.args

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        email_subject = f"{args.script} terminó correctamente"
        email_body = (f"El script {args.script} terminó correctamente.\n" +
                      "\n\t".join([f"{k}: {v}" for k, v in result.__dict__.items()]))
        print(email_body)
        send_email(email_subject, email_body)
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        email_subject = "{args.script} ERROR"
        email_body = (f"Error: The subroutine exited with non-zero status {e.returncode}.\n\n"
                        "Standard Output:\n" +
                        (e.stdout if e.stdout else "") +
                      "\nStandard Error:\n" +
                      (e.stderr if e.stderr else ""))
        print(email_body)
        send_email(email_subject, email_body)
        sys.exit(e.returncode)
    except Exception as e:
        email_subject = "Unexpected error in subroutine execution"
        email_body = f"An unexpected error occurred: {e}"
        print(email_body)
        send_email(email_subject, email_body)
        sys.exit(1)

if __name__ == "__main__":
    main()