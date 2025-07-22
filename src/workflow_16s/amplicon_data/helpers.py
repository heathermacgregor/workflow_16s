# =============================== HELPER FUNCTIONS ==================================== #
def _init_dict_level(a, b, c=None, d=None, e=None):
    if b not in a:
        a[b] = {}
    if c and c not in a[b]:
        a[b][c] = {}
    if d and d not in a[b][c]:
        a[b][c][d] = {}
    if e and e not in a[b][c][d]:
        a[b][c][d][e] = {}
