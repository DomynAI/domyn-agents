"""Whether a path is a link, junctions included.

Every place that refuses to follow a link asks here. ``is_symlink()`` alone misses a
Windows directory junction (``mklink /J``): it leads elsewhere exactly as a symbolic
link to a folder does, any user can make one without privileges, and Python reports
it as a plain directory (``is_symlink()`` and ``S_ISLNK`` false, ``is_dir`` true even
when not following links). ``.agents`` made a junction to the home folder would then
have loaded ``~/skills`` and opened it to the file tools, the case the link checks
exist for. Elsewhere there are no junctions and this is ``is_symlink()``.
"""

from __future__ import annotations

import os
from pathlib import Path


def is_link(path: Path | os.DirEntry[str]) -> bool:
    """A symbolic link or a junction: something that leads somewhere else."""
    return path.is_symlink() or path.is_junction()


def is_link_stat(result: os.stat_result) -> bool:
    """The same for an ``lstat`` result: a junction is a directory to ``S_ISLNK``,
    and only its reparse tag says what it is."""
    import stat

    if stat.S_ISLNK(result.st_mode):
        return True
    tag = getattr(result, "st_reparse_tag", 0)
    return bool(tag) and tag == getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", -1)
