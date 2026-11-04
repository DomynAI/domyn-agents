"""``/memory``: what the agent remembers about you, a page at a time.

The listing is plain lines (id, date, text), not the markdown the model is given:
the model needs structure it can read back, a person needs to scan. Pages are
numbered here, whatever cursor the manager uses underneath.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from domyn.agents.interfaces.cli.calls import as_written
from domyn.agents.memory.render import neutralize

if TYPE_CHECKING:
    from rich.console import Console

    from domyn.agents.interfaces.cli.theme import Theme
    from domyn.agents.memory.managers import BaseMemoryManager, MemoryPage

PAGE_SIZE = 20


async def fetch_page(
    manager: BaseMemoryManager, agent: Any, user_id: str, number: int
) -> tuple[MemoryPage, int, int]:
    """Page ``number`` (from 1), the number it really is, and the last page's.

    Cursors only go forward, so page 3 is reached through pages 1 and 2. A number
    past the end gives the last page, with its own number back.
    """
    cursor: str | None = None
    reached = 0
    while True:
        page = await manager.recall_page(user_id, agent, limit=PAGE_SIZE, cursor=cursor)
        reached += 1
        cursor = page.next_cursor
        if reached >= max(1, number) or cursor is None:
            break
    if page.total is not None:
        last = max(1, -(-page.total // PAGE_SIZE))
    else:
        last = reached + (1 if cursor else 0)
    return page, reached, last


def render(
    console: Console,
    theme: Theme,
    page: MemoryPage,
    *,
    number: int,
    last: int,
    path: str | None = None,
) -> None:
    """One page of memories, and how to read the next."""
    meta = theme.style("meta")
    if not page.items:
        console.print("  Nothing is remembered yet.", style=meta, highlight=False)
        if path:
            console.print(
                f"  file  {path}", style=meta, highlight=False, markup=False, soft_wrap=True
            )
        return
    console.print()
    for memory in page.items:
        # Escaped for a terminal, then cut: a memory is text somebody wrote, and an
        # escape sequence in one must not redraw the screen.
        text = as_written(neutralize(memory.memory))
        stamp = memory.timestamp.date().isoformat()
        console.print(
            f"  {memory.id:<8}  {stamp}  {text}", style=meta, highlight=False, markup=False
        )
    total = f" of {page.total}" if page.total is not None else ""
    more = f" · /memory list {number + 1} for the next page" if number < last else ""
    console.print(
        f"\n  page {number}/{last} · {len(page.items)}{total}{more}",
        style=meta,
        highlight=False,
        markup=False,
    )
    if path:
        console.print(f"  file  {path}", style=meta, highlight=False, markup=False, soft_wrap=True)
