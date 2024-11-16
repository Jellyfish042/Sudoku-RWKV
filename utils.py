def solve_sudoku_gt(board):
    def is_valid(grid, num, pos):
        # Row check
        for x in range(9):
            if grid[pos[0]][x] == num and pos[1] != x:
                return False
        
        # Column check
        for x in range(9):
            if grid[x][pos[1]] == num and pos[0] != x:
                return False
        
        # Box check
        box_x = pos[1] // 3
        box_y = pos[0] // 3
        for i in range(box_y * 3, box_y * 3 + 3):
            for j in range(box_x * 3, box_x * 3 + 3):
                if grid[i][j] == num and (i, j) != pos:
                    return False
        return True

    def is_valid_board(grid):
        for i in range(9):
            for j in range(9):
                if grid[i][j] != 0:
                    temp = grid[i][j]
                    grid[i][j] = 0
                    if not is_valid(grid, temp, (i, j)):
                        return False
                    grid[i][j] = temp
        return True

    def is_complete(grid):
        return all(all(cell != 0 for cell in row) for row in grid)

    def find_best_empty(grid):
        min_possibilities = 10
        best_pos = None
        
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    possibilities = sum(1 for num in range(1, 10) 
                                     if is_valid(grid, num, (i, j)))
                    if possibilities < min_possibilities:
                        min_possibilities = possibilities
                        best_pos = (i, j)
                        if possibilities == 1:
                            return best_pos
        return best_pos

    def solve(grid):
        nonlocal solutions_found, solution_grid
        if solutions_found > 1:
            return True
            
        pos = find_best_empty(grid)
        if not pos:
            if is_complete(grid):
                solutions_found += 1
                if solutions_found == 1:
                    for i in range(9):
                        for j in range(9):
                            solution_grid[i][j] = grid[i][j]
                return True
            return False
            
        row, col = pos
        valid_numbers = [num for num in range(1, 10) 
                        if is_valid(grid, num, (row, col))]
        
        for num in valid_numbers:
            grid[row][col] = num
            if solve(grid):
                if solutions_found > 1:
                    return True
            grid[row][col] = 0
            
        return False

    working_board = [row[:] for row in board]
    if not is_valid_board(working_board):
        return (0, None)
    
    solutions_found = 0
    solution_grid = [[0] * 9 for _ in range(9)]
    
    solve(working_board)
    
    if solutions_found == 0:
        return (0, None)
    if solutions_found > 1:
        return (2, None)
    return (1, solution_grid)
