"""
Finance LLM CLI using Typer + Rich + OpenAI
Location: src/lectures/day1-llms/finance_llm_cli.py
"""
from __future__ import annotations

import os
from typing import Optional

import typer
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

load_dotenv()  # Load environment variables from .env if present

console = Console()
app = typer.Typer(help="Finance LLM CLI — ask questions about markets")

DEFAULT_SYSTEM = (
    "You are a finance research assistant specialized in financial markets, "
    "asset pricing, time-series, and corporate finance. Be concise, show clear "
    "assumptions, provide brief step-by-step reasoning when helpful, and avoid "
    "unnecessary math notation."
)


def create_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print(
            Panel.fit(
                "Missing OPENAI_API_KEY. Set it in your environment or .env file.",
                title="Configuration error",
                style="bold red",
            )
        )
        raise typer.Exit(code=1)
    return OpenAI(api_key=api_key)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your finance question"),
    model: str = typer.Option("gpt-4o-mini", help="Model ID"),
    temperature: float = typer.Option(0.2, min=0.0, max=2.0, help="Sampling temperature"),
    max_tokens: int = typer.Option(500, help="Max new tokens for the answer"),
    system: Optional[str] = typer.Option(None, help="Override the finance system message"),
):
    """Ask a finance question and print a nicely formatted answer."""
    client = create_client()
    system_msg = system or DEFAULT_SYSTEM

    console.rule("Finance LLM")
    console.print(Panel.fit(question, title="Question", style="bold cyan"))

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
        ],
    )

    content = resp.choices[0].message.content
    console.print(Panel(Markdown(content), title="Answer", border_style="green"))

    if getattr(resp, "usage", None):
        usage = resp.usage
        table = Table(title="Token usage")
        table.add_column("prompt_tokens", justify="right")
        table.add_column("completion_tokens", justify="right")
        table.add_column("total_tokens", justify="right")
        table.add_row(
            str(getattr(usage, "prompt_tokens", "-")),
            str(getattr(usage, "completion_tokens", "-")),
            str(getattr(usage, "total_tokens", "-")),
        )
        console.print(table)


def main():
    try:
        app()
    except KeyboardInterrupt:
        console.print("\nInterrupted.")


if __name__ == "__main__":
    main()
