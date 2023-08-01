# How to Set Up a Virtual Environment in Visual Studio Code

Visual Studio Code (VS Code) is a free, open-source code editor that is lightweight yet powerful. One of its many features is the ability to manage Python virtual environments, which are tools for isolating project-specific dependencies from the global Python interpreter.

This guide will walk you through the process of setting up a virtual environment in VS Code.

## Prerequisites

Ensure you have the following installed:

1. Python: You can verify this by running `python --version` or `python3 --version` in your terminal or command prompt. 
2. Visual Studio Code: You can download it from [here](https://code.visualstudio.com/).
3. Python extension for Visual Studio Code: You can install it from within VS Code by clicking on the Extensions view icon on the Sidebar (or press `Ctrl+Shift+X`), searching for "Python", and clicking on `Install` for the one by Microsoft.

## Steps to Create a Virtual Environment

1. **Open the Terminal in VS Code**

   To open the terminal in VS Code, you can use the `Ctrl+` ` backtick shortcut, or navigate to `Terminal > New Terminal` from the menu. This will open a new terminal at the bottom of your VS Code window.

   <!-- Screenshot 1: Opening terminal in VS Code -->

2. **Navigate to Your Project Directory**

   Use the `cd` command followed by the path of your project directory to navigate into it. For example, if your project directory is named "my_project" and is located in your "Documents" folder, you could use the following command:

   ```bash
   cd Documents/my_project
   ```
   
   <!-- Screenshot 2: Navigating to project directory in terminal -->

3. **Create the Virtual Environment**

   Once you're in your project directory, you can create a new virtual environment using the `venv` module that comes with Python. The following command creates a new virtual environment named "venv" in your project directory:

   ```bash
   python3 -m venv venv
   ```
   
   If you're using Windows, use `python` instead of `python3`.  if your using linux 

   <!-- Screenshot 3: Creating virtual environment -->

4. **Activate the Virtual Environment**

   The method to activate your virtual environment depends on your operating system:

   - On macOS and Linux:

     ```bash
     source venv/bin/activate
     ```

   - On Windows:

     ```bash
     .\venv\Scripts\activate
     ```

   Once the virtual environment is activated, you'll see `(venv)` at the beginning of your command prompt.

   <!-- Screenshot 4: Activating virtual environment -->

5. **Set the Python Interpreter in VS Code**

   After creating and activating your virtual environment, you need to tell VS Code to use the Python interpreter in this environment. You can do this by clicking on the Python version in the bottom-left corner of VS Code (or using `Ctrl+Shift+P` and searching for "Python: Select Interpreter"), then selecting the interpreter that's located in your project directory under the "venv" folder.

   <!-- Screenshot 5: Setting Python interpreter in VS Code -->

6. **Install Packages as Needed**

   Now that your virtual environment is set up and activated, you can install any necessary packages using `pip`. For example, to install the requests package, you would use the following command:

   ```bash
   pip install requests
   ```
   
   <!-- Screenshot 6: Installing packages -->

And that's it! You now have a Python virtual environment set up in VS Code, isolated from your global Python interpreter. This allows you to manage your project's dependencies more effectively.

Remember to always activate your virtual environment before you start working on your project, and deactivate it when you're done. You can deactivate the environment with the following command:

```bash
deactivate
```

<!-- Screenshot 7: Deactivating virtual environment -->

Happy coding!
