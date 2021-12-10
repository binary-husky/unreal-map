from pympler import tracker
tr = tracker.SummaryTracker()

def memdb_print_diff():
    tr.print_diff()

    