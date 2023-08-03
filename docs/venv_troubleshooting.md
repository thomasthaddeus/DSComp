# Troubleshooting your Virtual Environment in VSCode
This troubleshooting guide is predicated on the member already following the 'venv_setup.md' walkthrough also located in the 'docs' folder of this repo

## Prerequisites

Ensure you have the following installed:

1. Python: You can verify this by running `python --version` or `python3 --version` in your terminal or command prompt. 
2. Visual Studio Code: You can download it from [here](https://code.visualstudio.com/).
3. Python extension for Visual Studio Code: You can install it from within VS Code by clicking on the Extensions view icon on the Sidebar (or press `Ctrl+Shift+X`), searching for "Python", and clicking on `Install` for the one by Microsoft.

## Setting Execution Policy

To change the PowerShell execution policy on your Windows computer, use the Set-ExecutionPolicy cmdlet. The change is effective immediately. You don't need to restart PowerShell.

> **NOTE:**
> In Windows Vista and later versions of Windows, to run commands that change the execution policy for the local computer, LocalMachine scope, start PowerShell with the Run as administrator option.

To change your execution policy:

PowerShell

Copy
Set-ExecutionPolicy -ExecutionPolicy <PolicyName>
For example:

PowerShell

Copy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned

## Error Codes when activating Virtual Environment in Windows OS

  1. **Error Code 1**
     Here is an image of an error recieved when attempting to activate the Virtual Environment
      ![errorcode1](https://github.com/TheMightiestCaz/DS-Competition/assets/115377584/3297b24c-3930-4dc1-a767-f2f4a13259f4)
     When researching this code online, you will find that it means your 'Execution Policy' is not set correctly

  2. **Next Error Code**

  
   <!-- Screenshot 1: Opening terminal in VS Code -->



Happy coding!
