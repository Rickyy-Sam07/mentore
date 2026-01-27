@echo off
echo ======================================================================
echo BERT CHAPTER CLASSIFICATION - QUICK START
echo ======================================================================
echo.

echo Step 1: Installing missing dependencies...
pip install matplotlib seaborn

echo.
echo Step 2: Testing environment...
python test_setup.py

echo.
echo ======================================================================
echo SETUP COMPLETE!
echo ======================================================================
echo.
echo To train the model, run:
echo     python train.py
echo.
echo To make predictions after training, run:
echo     python predict.py
echo.
echo NOTE: You currently have PyTorch CPU version installed.
echo For faster training with your RTX 3050 GPU, see INSTALLATION.md
echo.
pause
