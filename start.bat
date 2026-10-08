@echo off
REM One-click launcher: starts Ollama (if needed) and the Streamlit chat app.
cd /d "%~dp0"
set OLLAMA_MODELS=D:\Coding\.ollama\models
set OLLAMA_CONTEXT_LENGTH=8192
set PYTHONWARNINGS=ignore
REM Cap math-library threads: each thread reserves memory, which fails on a busy 16 GB machine
set OPENBLAS_NUM_THREADS=4
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4

curl -s http://localhost:11434/api/version >nul 2>&1
if errorlevel 1 (
    echo Starting Ollama...
    start "Ollama" /min "D:\Coding\Ollama\ollama.exe" serve
    timeout /t 4 /nobreak >nul
)

echo Starting the chat app at http://localhost:8501  (close this window to stop it)
.venv\Scripts\streamlit.exe run app.py
