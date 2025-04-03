# /// script
# dependencies = [
#   "llm>=0.23",
#   "llm-anthropic",
#   "llm-gemini>=0.16",
#   "rich>=13.0.0",
#   "typer>=0.15.2",
# ]
# ///

"""Make file-level edits with an LLM."""

import os
import subprocess
import tempfile
from typing import Annotated

import llm
import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="Edit files using LLMs.")

CODE_CMD = "code-oss-cloud-workstations"


@app.command()
def main(
    file_path: Annotated[
        str,
        typer.Argument(
            ...,
            help="Path to the file to edit",
            exists=True,
        ),
    ],
    instruction: Annotated[
        str,
        typer.Option(
            ...,
            "--instruction",
            "-i",
            prompt=True,
            help="Instruction to use for generating edits",
        ),
    ],
    model: Annotated[
        str,
        typer.Option(
            ...,
            "--model",
            "-m",
            help="Model to use for generating edits",
        ),
    ] = "gemini-2.0-flash",
):
    r"""
    Edit files using LLMs.

    ---------------------\n\n
    * Loads a file and asks for your editing instructions\n
    * Applies the instructions in a temporary copy\n
    * Loads the diff for inspection\n
    * Asks for approval [yN]\n
    * If approved makes the edits to the file.\n\n

    Example\n
    -------\n
    $ uv run scripts/llmedit.py path/to/file.py
    """
    # Read the input file
    try:
        with open(file_path, "r") as f:
            original_content = f.read()
    except Exception as e:
        console.print(f"[bold red]Error reading file:[/bold red] {e}")
        return

    # Get the instruction from the user
    # console.print("> Add your instruction: ", end="")
    # instruction = input()

    # Construct the prompt
    system_prompt = f"""
    # Main Instruction
    {instruction}

    # Extra instructions
    * Apart from the changes you've been instructed to make, rewrite the file faithfully
    and completely
    * Do not produce any output apart from the file
    * Your output will be piped directly into the result, so don't add any other explanation, just unfenced code.
    * IMPORTANT: Do not add python fenced codeblocks!
    * IMPORTANT: Do not miss any # commented instructions at the top of the file!
    """

    try:
        # Get the LLM model
        llm_model = llm.get_model(model)

        # Create a loading message
        with console.status(
            f"[bold green]Processing with {model}...[/bold green]"
        ):
            # Execute the prompt
            response = llm_model.prompt(original_content, system=system_prompt)
            edited_content = response.text().strip()
    except Exception as e:
        console.print(f"[bold red]Error processing with LLM:[/bold red] {e}")
        return

    # Create a temporary file with the edited content
    temp_file = tempfile.NamedTemporaryFile(
        suffix=os.path.splitext(file_path)[1], mode="w", delete=False
    )
    temp_file_path = temp_file.name
    try:
        temp_file.write(edited_content)
        temp_file.close()

        # Show diff using VS Code
        console.print(
            "\n[bold]Opening diff in VS Code...(close to continue)[/bold]"
        )
        try:
            subprocess.run(
                [CODE_CMD, "--wait", "--diff", file_path, temp_file_path],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            console.print(
                f"[bold red]Error opening VS Code diff:[/bold red] {e}"
            )
            os.unlink(temp_file_path)
            return
        except FileNotFoundError:
            console.print(
                "[bold red]Error: 'code' command not found. Make sure VS Code is installed and in your PATH.[/bold red]"
            )
            os.unlink(temp_file_path)
            return

        # Ask for confirmation
        console.print("\n> Do you want to accept these changes? [yN] ", end="")
        choice = input().lower()

        if choice == "y" or choice == "yes":
            try:
                with open(file_path, "w") as f:
                    f.write(edited_content)
                console.print(
                    f"[bold green]> {file_path} edited.[/bold green]"
                )
            except Exception as e:
                console.print(
                    f"[bold red]Error writing to file:[/bold red] {e}"
                )
        else:
            console.print("[yellow]Changes discarded.[/yellow]")
    finally:
        os.unlink(temp_file_path)


if __name__ == "__main__":
    app()
