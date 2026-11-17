"""Prep: the stages that turn a donation into a corpus. Runs here so a bug is a rebuild, not a re-upload.

`render.py` re-renders the message format from bundle fields; `parity.py` proves it matches the cache
the donor tool wrote; `merge.py` re-merges the parts a donation kept; `canonical.py` writes the bundle
the retrieval side reads.
"""
