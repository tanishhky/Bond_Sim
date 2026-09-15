# 0002: Full ALFRED vintages from day one, with an explicit pre-vintage rule

**Decision (2026-09-15, Tanishk's call).** Every FRED series is pulled as its
complete revision history (`output_type=1`, the whole real-time axis) and
stored as a `VintageFrame`. Any analysis "as of" a date sees exactly the
vintage that was current then. The no-lookahead invariance test (`tests/
test_no_lookahead.py`) proves that appending later data cannot change an
earlier as_of result.

**Why the strict option.** The thesis makes lead/lag and regime claims
("debt/GDP changes lead term-premium changes by N quarters"). On revised data
those claims can be artifacts: a 2009 GDP print was revised five times
(14,301.5 initial to 14,448.9 today), and employment is benchmark-revised
every year. A reviewer who asks "did anyone actually know this at the time?"
must get a yes.

**Two facts about ALFRED that forced design choices.**

1. *Per-request cap of 2000 vintage dates.* Daily series accumulate one
   vintage date per trading day (DGS10: 5,108 since 2005-06-28). The client
   chunks the real-time axis by vintage-date windows and coalesces adjacent
   windows that carry the same value (`data/fred.py::_coalesce`).

2. *Vintages begin when ALFRED started tracking the series.* GDP: 1991-12-04.
   DGS10: 2005-06-28. UNRATE and most BLS series: mid-1990s. Every older
   observation carries the first vintage date as `realtime_start`, which is
   not when it was first published, only when ALFRED first recorded it.
   Left alone, an as_of view for 1985 would be empty for every series.

   **Pre-vintage rule.** For rows whose `realtime_start` equals the series'
   first vintage date and whose observation predates it by more than the
   series' `pub_lag_days`, `realtime_start` is set to `date + pub_lag_days`
   and the row is flagged `backfilled=True`. The value used is the
   first-vintage value (already revised relative to the true first print;
   nothing earlier exists). `pub_lag_days` are conservative, i.e. later than
   the typical release, so a backfilled timestamp can only understate what
   was known, never overstate it. `FredClient.vintage_coverage()` produces
   the table the paper's data section must include: first true vintage and
   backfilled share per series.

**Consequence for the paper.** Point-in-time results are exact where true
vintages exist and approximate (first-vintage value, assumed lag) before
that. Any lead/lag claim whose sample is mostly pre-vintage must say so.

**Alternative rejected.** RateWalk's `output_type=4` (initial release only):
sufficient for a one-series feed, but it cannot answer "what did the whole
panel look like on date X" and it hits the same vintage-date cap on daily
series.
