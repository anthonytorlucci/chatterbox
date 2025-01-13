import argparse
import subprocess

def main():
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Run Streamlit chat applications")

    # Add chat argument
    parser.add_argument("--chat", action="store_true", help="Run basic chat application")
    parser.add_argument("--summary", action="store_true", help="Run research context summary application")

    # Parse arguments
    args = parser.parse_args()

    # Run the shell command if --chat is provided
    if args.chat:
        subprocess.run("uv run --env-file .env -- streamlit run st_chat.py", shell=True)

    if args.summary:
        subprocess.run("uv run --env-file .env -- streamlit run st_summary.py", shell=True)

    # coming soon -> researcher

if __name__ == "__main__":
  main()
