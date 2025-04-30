"""Console script for cocox."""
import cocox

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for cocox."""
    console.print("Replace this message by putting your code into "
               "cocox.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
