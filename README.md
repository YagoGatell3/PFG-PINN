# Bachelor Thesis Python Template

This repository serves as a Python template for Bachelor Thesis projects. Follow the steps below to set up your environment and start working on your thesis.

---

## Prerequisites

Ensure you have the following tools installed:

### 1. Install Python

- Download and install Python from the official website: [Python Downloads](https://www.python.org/downloads/)

### 2. Install Visual Studio Code (VSCode)

- Download and install VSCode from the official website: [VSCode Downloads](https://code.visualstudio.com/). Then, set up VSCode with GitHub Copilot [for free](https://code.visualstudio.com/docs/copilot/setup-simplified).

### 3. Install VSCode extensions

Install the following extensions in VSCode:

- [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance): Pylance provides rich language support for Python, including auto-completions, type checking, and code navigation.

- [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter): Jupyter allows you to run Jupyter notebooks directly within VSCode, which is great for data analysis and exploratory programming.

- [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff): Ruff is a fast Python linter that enforces coding standards and helps catch bugs early.

- [Autodocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring): This extension helps generate Python docstrings automatically, saving time and ensuring consistency.

- [Rainbow Indent](https://marketplace.visualstudio.com/items?itemName=oderwat.indent-rainbow): Indent Rainbow highlights indentation levels with different colors, making it easier to spot mistakes.

- [Python Indent](https://marketplace.visualstudio.com/items?itemName=KevinRose.vsc-python-indent): This extension improves Python indentation in VSCode, making coding more intuitive and efficient.

- [Better Comments](https://marketplace.visualstudio.com/items?itemName=aaron-bond.better-comments): Better Comments makes your comments more readable by adding different styles for different types of comments (e.g., notes, warnings).

- [IntelliCode](https://marketplace.visualstudio.com/items?itemName=VisualStudioExptTeam.vscodeintellicode): IntelliCode provides AI-assisted code recommendations based on common coding patterns.

To install extensions:

1. Open VSCode.
2. Go to the Extensions view by clicking on the Extensions icon in the Activity Bar or pressing `Ctrl+Shift+X`.
3. Search for each extension by name and click "Install".

---

## Setting up your virtual environment

### 1. Install `uv`

`uv` is a Python environment management tool that simplifies creating, activating, and maintaining virtual environments, along with managing dependencies. To install it, simply run this command:

```bash
pip install uv
```

### 2. Create and activate an Environment Using `uv`

1. Create an environment named `env` using Python 3.13:

   ```bash
   uv venv env --python 3.13
   ```

2. Activate the environment:

     ```bash
     source .\env\Scripts\activate
     ```

### 3. Updating dependencies with `uv`

The dependencies in the project are defined in `pyproject.toml`, in the `[dependencies]`. To add or modify the dependencies of the project:

1. Open the `pyproject.toml` file in a text editor.
2. Add or modify the dependencies under the `[tool.dependencies]` section. For example:

   ```toml
   [project]
   dependencies = ["numpy", "pandas"]
   ```

3. After editing, run:

   ```bash
   uv sync
   ```

    This will ensure your environment is synchronized with the latest versions of required dependencies. In addition, this command updates  the `uv.lock` file, which locks the exact versions of dependencies installed in your environment and ensures consistency across different setups by freezing the dependency tree.

### 4. Freezing dependencies with `uv`

To see the current state of your dependencies in the console, run

```bash
uv pip freeze
```

This command outputs the exact versions of installed packages, which is helpful for reproducibility.

### 5. Running Python Scripts with `uv`

You can execute a Python script within your environment using the `uv run` command:

```bash
uv run src/prueba.py
```

This ensures the script is run with the correct environment and dependencies.

### 6. Selecting the Python interpreter in VSCode

1. Open your project in VSCode.
2. Press `Ctrl+Shift+P` to open the Command Palette.
3. Type `Python: Select Interpreter` and select it.
4. Choose the Python interpreter associated with your `env` environment (it should have `env` in its path).

---

## Updating the project to GitHub

1. If you have not already initialized a Git repository for your project, run:

    ```bash
    git init
    ```

2. Stage all changes:

    ```bash
    git add .
    ```

3. Commit the staged files with a message:

    ```bash
    git commit -m "Initial commit"
    ```

4. Create a new repository on GitHub.
5. Link your local repository to the remote repository:

   ```bash
   git remote add origin <repository-url>.git
   ```

6. Push the committed changes to GitHub:

    ```bash
    git branch -M main
    git push -u origin main
    ```

---

Feel free to modify this template as needed for your thesis project. If you encounter any issues, refer to the documentation or reach out to your supervisor.
