set root=C:\ProgramData\Anaconda3
call %root%\Scripts\activate.bat %root%
start "Data Gathering" cmd /k python -W ignore "Data Gatherer.py"