@echo off
echo Activating virtual environment...
call "%~dp0venv\Scripts\activate.bat"

echo Changing to project directory...
cd /d "%~dp0MED_AI_PFA"

echo Starting Django development server...
python manage.py runserver

pause