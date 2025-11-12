@echo off
echo Setting up Python virtual environment for LLM Grader...

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo Creating .env file from template...
    copy .env.example .env
    echo Please edit .env file and add your OpenAI API key
) else (
    echo .env file already exists
)

echo.
echo Setup complete! 
echo.
echo Next steps:
echo 1. Edit .env file and add your OpenAI API key
echo 2. Run: python grade_conversations.py
echo.
pause