class Node:
    def __init__(self, row=None, is_column=False, name=None):
        self.left = self
        self.right = self
        self.up = self
        self.down = self
        self.column = self if is_column else None
        self.row = row
        self.size = 0          # only meaningful for column headers
        self.name = name       # column header name (optional)


def build_dlx(matrix):
    num_rows = len(matrix)
    if num_rows == 0:
        return None
    num_cols = len(matrix[0])

    header = Node(is_column=True, name="header")
    columns = []

    # create column headers
    for col in range(num_cols):
        column = Node(is_column=True, name=col)

        column.right = header
        column.left = header.left
        header.left.right = column
        header.left = column

        columns.append(column)

    # create data nodes
    for row_idx, row in enumerate(matrix):
        row_nodes = []
        for col_idx, val in enumerate(row):
            if val == 1:
                col_node = columns[col_idx]
                node = Node(row=row_idx)
                node.column = col_node

                # insert into column (at bottom, just above header)
                node.down = col_node
                node.up = col_node.up
                col_node.up.down = node
                col_node.up = node

                col_node.size += 1
                row_nodes.append(node)

        # link row nodes circularly
        if row_nodes:
            L = len(row_nodes)
            for i in range(L):
                row_nodes[i].left = row_nodes[(i - 1) % L]
                row_nodes[i].right = row_nodes[(i + 1) % L]

    return header


def cover(col_node):
    # remove column header
    col_node.right.left = col_node.left
    col_node.left.right = col_node.right

    row = col_node.down
    while row != col_node:
        right_node = row.right
        while right_node != row:
            right_node.down.up = right_node.up
            right_node.up.down = right_node.down
            right_node.column.size -= 1
            right_node = right_node.right
        row = row.down


def uncover(col_node):
    row = col_node.up
    while row != col_node:
        left_node = row.left
        while left_node != row:
            left_node.column.size += 1
            left_node.down.up = left_node
            left_node.up.down = left_node
            left_node = left_node.left
        row = row.up

    col_node.right.left = col_node
    col_node.left.right = col_node


def select_column(header):
    # choose the column with minimum size (Algorithm X heuristic)
    min_size = float("inf")
    selected = None
    col = header.right
    while col != header:
        if col.size < min_size:
            min_size = col.size
            selected = col
        col = col.right
    return selected


def dlx_search(header, stats):
    """
    Returns True if a solution is found, else False.

    stats:
      - conflicts: number of times we encounter a dead-end (selected column has size 0)
      - propagations: number of forced choices (selected column has size 1)
    """
    if header.right == header:
        return True

    col = select_column(header)

    # conflict: dead end
    if col.size == 0:
        stats["conflicts"] += 1
        return False

    # propagation: forced move (unit-like)
    if col.size == 1:
        stats["propagations"] += 1

    cover(col)

    row = col.down
    while row != col:
        # cover columns of the chosen row
        right_node = row.right
        while right_node != row:
            cover(right_node.column)
            right_node = right_node.right

        if dlx_search(header, stats):
            return True

        # undo covers for this row
        left_node = row.left
        while left_node != row:
            uncover(left_node.column)
            left_node = left_node.left

        row = row.down

    uncover(col)
    return False


def solve(matrix):
    """
    Solve exact cover with DLX (find one solution).
    Returns:
      found: bool
      conflicts: int
      propagations: int
    """
    header = build_dlx(matrix)
    if header is None:
        return False, 0, 0

    stats = {"conflicts": 0, "propagations": 0}
    found = dlx_search(header, stats)
    return found, stats["conflicts"], stats["propagations"]