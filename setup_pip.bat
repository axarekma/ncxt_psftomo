@REM @echo off 
@REM del build\lib.win-amd64-cpython-310\_projectors.cp310-win_amd64.pyd
@REM del _projectors.cp310-win_amd64.pyd

@REM python setup.py build_ext --inplace
@REM if NOT %ERRORLEVEL% == 0 goto :endofscript 

pip install --upgrade ./
python -m unittest -v

@REM python test_omp.py

:endofscript
if %ERRORLEVEL% == 0 echo "Everything Went fine!"
if NOT %ERRORLEVEL% == 0 (
    echo "Something Failed, ABORT."
    )   