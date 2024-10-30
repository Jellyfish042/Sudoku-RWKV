from colorama import init, Fore, Style
import textwrap


def print_sudoku(grid):
    """
    Prints a sudoku grid in a beautiful format with aligned numbers.
    
    Args:
        grid: 9x9 list of lists containing the sudoku numbers (0 represents empty cells)
    """
    # ANSI escape codes for colors
    BLUE = '\033[94m'
    RESET = '\033[0m'
    
    # Define the horizontal lines
    thick_line = BLUE + "╔═══╤═══╤═══╦═══╤═══╤═══╦═══╤═══╤═══╗" + RESET
    thin_line =  BLUE + "╟───┼───┼───╫───┼───┼───╫───┼───┼───╢" + RESET
    mid_line =   BLUE + "╠═══╪═══╪═══╬═══╪═══╪═══╬═══╪═══╪═══╣" + RESET
    bottom_line = BLUE + "╚═══╧═══╧═══╩═══╧═══╧═══╩═══╧═══╧═══╝" + RESET

    print("\n" + thick_line)
    
    for i in range(9):
        # Print the row with vertical separators
        row = BLUE + "║" + RESET
        for j in range(9):
            # Add the number (with appropriate spacing)
            if grid[i][j] == 0:
                row += " · "  # Display dots for empty cells
            else:
                row += f" {grid[i][j]} "  # Add space after number for alignment
                
            # Add vertical separators
            if j < 8:
                if (j + 1) % 3 == 0:
                    row += BLUE + "║" + RESET
                else:
                    row += BLUE + "│" + RESET
            else:
                row += BLUE + "║" + RESET
                
        print(row)
        
        # Print horizontal lines
        if i < 8:
            if (i + 1) % 3 == 0:
                print(mid_line)
            else:
                print(thin_line)
        else:
            print(bottom_line)


def print_sudoku_comparison(grid1, grid2, title1="Original", title2="Solved", highlight_diff=True):
    """
    Prints two sudoku grids side by side for comparison with perfect alignment.
    
    Args:
        grid1: First 9x9 grid to display
        grid2: Second 9x9 grid to display
        title1: Title for the first grid (default: "Original")
        title2: Title for the second grid (default: "Solved")
        highlight_diff: If True, highlights differences between grids in red on both sides (default: True)
    """
    # ANSI escape codes for colors
    BLUE = '\033[94m'
    RED = '\033[91m'
    RESET = '\033[0m'
    
    # Define the horizontal lines
    thick_line = BLUE + "╔═══╤═══╤═══╦═══╤═══╤═══╦═══╤═══╤═══╗" + RESET
    thin_line = BLUE + "╟───┼───┼───╫───┼───┼───╫───┼───┼───╢" + RESET
    mid_line = BLUE + "╠═══╪═══╪═══╬═══╪═══╪═══╬═══╪═══╪═══╣" + RESET
    bottom_line = BLUE + "╚═══╧═══╧═══╩═══╧═══╧═══╩═══╧═══╧═══╝" + RESET

    # Calculate padding for centering titles
    grid_width = 38
    title1_padding = (grid_width - len(title1)) // 2
    title2_padding = (grid_width - len(title2)) // 2
    
    # Print centered titles
    print("\n" + " " * title1_padding + title1 + 
          " " * (grid_width - len(title1) - title1_padding + 4) +
          " " * title2_padding + title2)
    
    print(" " + thick_line + "    " + thick_line)
    
    def format_cell(num):
        """Format a single cell with proper spacing"""
        if num == 0:
            return " · "  # centered dot for empty cells
        return f" {num} "  # centered number
    
    for i in range(9):
        # Print both rows side by side
        row1 = BLUE + "║" + RESET
        row2 = BLUE + "║" + RESET
        
        for j in range(9):
            # Format cells for both grids
            cell1 = format_cell(grid1[i][j])
            cell2 = format_cell(grid2[i][j])
            
            # Add highlighting if cells are different
            if highlight_diff and grid1[i][j] != grid2[i][j]:
                row1 += RED + cell1 + RESET  # Highlight first grid
                row2 += RED + cell2 + RESET  # Highlight second grid
            else:
                row1 += cell1
                row2 += cell2
            
            # Add vertical separators without extra spaces
            if j < 8:
                if (j + 1) % 3 == 0:
                    row1 += BLUE + "║" + RESET
                    row2 += BLUE + "║" + RESET
                else:
                    row1 += BLUE + "│" + RESET
                    row2 += BLUE + "│" + RESET
            else:
                row1 += BLUE + "║" + RESET
                row2 += BLUE + "║" + RESET
        
        print(" " + row1 + "    " + row2)
        
        # Print horizontal lines
        if i < 8:
            if (i + 1) % 3 == 0:
                print(" " + mid_line + "    " + mid_line)
            else:
                print(" " + thin_line + "    " + thin_line)
        else:
            print(" " + bottom_line + "    " + bottom_line)
            
            
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


def extract_between(text: str, start_str: str, end_str: str, include_boundaries: bool = False) -> list[str]:
    if not text or not start_str or not end_str:
        return []
    
    results = []
    current_pos = 0
    
    while True:
        start_pos = text.find(start_str, current_pos)
        if start_pos == -1:
            break
            
        end_pos = text.find(end_str, start_pos + len(start_str))
        if end_pos == -1:
            break
            
        if include_boundaries:
            extracted = text[start_pos:end_pos + len(end_str)]
        else:
            extracted = text[start_pos + len(start_str):end_pos]
            
        results.append(extracted)
        current_pos = end_pos + len(end_str)
    
    return results


# def print_detected_error(mp, gt):
#     print(f'\n{f" Error detected ":=^100}')
#     print(f' - Model Prediction:\n{mp}')
#     print('-' * 100)
#     print(f' - Ground Truth:\n{gt}')
#     print('=' * 100)

def print_detected_error(mp, gt, width=100):
    init()
    
    border_char = "═"
    separator = "─"
    
    def create_header(text, width=width):
        padding = (width - len(text) - 2) // 2
        return f"{border_char * padding} {text} {border_char * (width - padding - len(text) - 2)}"
    
    def format_content(text, indent=4):
        text_str = str(text)
        lines = text_str.split('\n')
        formatted_lines = []
        
        for line in lines:
            if line.strip():
                wrapped = textwrap.fill(
                    line,
                    width=width-indent,
                    initial_indent=" "*indent,
                    subsequent_indent=" "*indent,
                    break_long_words=True,
                    replace_whitespace=False
                )
                formatted_lines.append(wrapped)
            else:
                formatted_lines.append(" "*indent)
                
        return '\n'.join(formatted_lines)
    
    print(f"\n{Fore.RED}{Style.BRIGHT}{create_header('Error detected')}{Style.RESET_ALL}")
    
    print(f"{Fore.CYAN}▶ Model Prediction:{Style.RESET_ALL}")
    print(format_content(mp))
    print(f"{Fore.BLUE}{separator * width}{Style.RESET_ALL}")
    
    print(f"{Fore.GREEN}▶ Ground Truth:{Style.RESET_ALL}")
    print(format_content(gt))
    print(f"{Fore.RED}{border_char * width}{Style.RESET_ALL}")