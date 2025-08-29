class Node:
    def __init__(self, row=None, is_column=False, name=None):
        self.left = self
        self.right = self
        self.up = self
        self.down = self
        self.column = self if is_column else None
        self.row = row
        self.size = 0  # 仅列头节点使用
        self.name = name  # 列头节点的名称

def build_dlx(matrix):
    num_rows = len(matrix)
    if num_rows == 0:
        return None
    num_cols = len(matrix[0])
    header = Node(is_column=True, name="header")
    columns = []
    
    # 创建列头节点
    for col in range(num_cols):
        column = Node(is_column=True, name=col)
        column.right = header
        column.left = header.left
        header.left.right = column
        header.left = column
        columns.append(column)
    
    # 创建数据节点并连接
    for row_idx in range(num_rows):
        row = matrix[row_idx]
        row_nodes = []
        for col_idx, val in enumerate(row):
            if val == 1:
                col_node = columns[col_idx]
                node = Node(row=row_idx)
                node.column = col_node
                node.down = col_node
                node.up = col_node.up
                col_node.up.down = node
                col_node.up = node
                col_node.size += 1
                row_nodes.append(node)
        
        # 连接行内节点
        if row_nodes:
            for i in range(len(row_nodes)):
                node = row_nodes[i]
                node.left = row_nodes[(i - 1) % len(row_nodes)]
                node.right = row_nodes[(i + 1) % len(row_nodes)]
    
    return header

def cover(col_node):
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
    min_size = float('inf')
    selected_col = None
    col = header.right
    while col != header:
        if col.size < min_size:
            min_size = col.size
            selected_col = col
        col = col.right
    return selected_col

def dlx_search(header, solution, backtrack_count):
    if header.right == header:
        return True
    col = select_column(header)
    cover(col)
    row = col.down
    while row != col:
        solution.append(row.row)
        right_node = row.right
        while right_node != row:
            cover(right_node.column)
            right_node = right_node.right
        
        if dlx_search(header, solution, backtrack_count):
            return True
        
        backtrack_count[0] += 1
        solution.pop()
        left_node = row.left
        while left_node != row:
            uncover(left_node.column)
            left_node = left_node.left
        
        row = row.down
    uncover(col)
    return False

def solve(matrix):
    header = build_dlx(matrix)
    if not header:
        return False, 0
    solution = []
    backtrack_count = [0]
    found = dlx_search(header, solution, backtrack_count)
    return found, backtrack_count[0]
