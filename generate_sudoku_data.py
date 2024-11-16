import random
from copy import deepcopy
from tqdm import tqdm
import json
import multiprocessing as mp
from functools import partial
import os
import time
from utils import *
from formatter import *


class Logger:
    def __init__(self, print_to_console=True):
        self.log = ""
        self.print_to_console = print_to_console

    def print_and_log(self, text: str, end="\n"):
        if self.print_to_console:
            print(text)
        self.log += text + end

    def print_all(self, max_length=2000):
        print("=" * 50)
        if max_length is None:
            print(self.log)
        else:
            print(self.log[:max_length])
        print("=" * 50)
        print(f"Total Length: {len(self.log)}")

    def clear(self):
        self.log = ""

    def append_to_jsonl(self, filename: str):
        with open(filename, "a", encoding="utf-8") as f:
            json_entry = json.dumps({"text": self.log.strip()}, ensure_ascii=False)
            f.write(json_entry + "\n")


def is_valid(grid, row, col, num):
    if num in grid[row]:
        return False

    if num in (grid[i][col] for i in range(9)):
        return False

    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if grid[i + start_row][j + start_col] == num:
                return False

    return True


def find_empty(grid):
    for i in range(9):
        for j in range(9):
            if grid[i][j] == 0:
                return i, j
    return None


def solve_grid(grid):
    empty = find_empty(grid)
    if not empty:
        return True
    row, col = empty
    for num in random.sample(range(1, 10), 9):
        if is_valid(grid, row, col, num):
            grid[row][col] = num
            if solve_grid(grid):
                return True
            grid[row][col] = 0
    return False


def count_solutions(grid, limit=2):
    empty = find_empty(grid)
    if not empty:
        return 1
    row, col = empty
    count = 0
    for num in range(1, 10):
        if is_valid(grid, row, col, num):
            grid[row][col] = num
            count += count_solutions(grid, limit - count)
            grid[row][col] = 0
            if count >= limit:
                break
    return count


def generate_sudoku_base():
    grid = [[0 for _ in range(9)] for _ in range(9)]
    for i in range(0, 9, 3):
        nums = random.sample(range(1, 10), 9)
        for r in range(3):
            for c in range(3):
                grid[i + r][i + c] = nums[r * 3 + c]
    return grid


def generate_sudoku(difficulty, seed=None):
    if seed is not None:
        random.seed(seed)
    else:
        random.seed()
    grid = generate_sudoku_base()

    solve_grid(grid)

    solved_grid = deepcopy(grid)

    cells = [(i, j) for i in range(9) for j in range(9)]
    random.shuffle(cells)

    for i, j in cells:
        if sum(row.count(0) for row in grid) >= difficulty:
            break
        temp = grid[i][j]
        grid[i][j] = 0
        if count_solutions(deepcopy(grid)) != 1:
            grid[i][j] = temp

    return grid, solved_grid


class Sudoku:
    def __init__(self, grid):
        self.grid = deepcopy(grid)
        self.possible_value_matrix = [[9 for _ in range(9)] for _ in range(9)]
        self.update_possible_value_matrix()

    def update_possible_value_matrix(self, add_noise=False):
        for i in range(9):
            for j in range(9):
                if self.grid[i][j] != 0:
                    self.possible_value_matrix[i][j] = 0
                else:
                    self.possible_value_matrix[i][j] = self.estimate_possible_values_v2(i, j)
                    # self.possible_value_matrix[i][j] = self.estimate_possible_values(i, j)
                    
                    # if add_noise:
                    #     self.possible_value_matrix[i][j] += random.randint(-2, 2)

    def estimate_possible_values_v2(self, row, col):
        possible_values = set(range(1, 10))

        # Check row
        possible_values -= set(self.grid[row])

        # Check column
        possible_values -= set(self.grid[i][col] for i in range(9))

        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if self.grid[i][j] != 0:
                    possible_values.discard(self.grid[i][j])

        return max(len(possible_values), 1)  # set minimum possible values to 1 to avoid exit when sudoku is not solved

    def estimate_possible_values(self, row, col):
        if self.grid[row][col] != 0:
            return 0

        # Count filled cells in the row
        row_filled = sum(1 for value in self.grid[row] if value != 0)

        # Count filled cells in the column
        col_filled = sum(1 for i in range(9) if self.grid[i][col] != 0)

        # Count filled cells in the 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        box_filled = sum(1 for i in range(box_row, box_row + 3) for j in range(box_col, box_col + 3) if self.grid[i][j] != 0)

        # Take the maximum of filled cells in row, column, and box
        max_filled = max(row_filled, col_filled, box_filled)

        # Estimate possible values
        possible_values = 9 - max_filled
        return max(1, possible_values)  # Ensure at least 1 possible value

    def find_min_possible_value_position(self):
        min_value = 10
        min_position = (-1, -1)

        for i in range(9):
            for j in range(9):
                current_value = self.possible_value_matrix[i][j]
                if 0 < current_value < min_value:
                    min_value = current_value
                    min_position = (i, j)

        if min_value == 10:
            return None
        return min_position
    
    def is_filled(self):
        return all(all(cell != 0 for cell in row) for row in self.grid)


def solve_sudoku(sudoku, logger):
    logger.print_and_log(f"<input>\n{format_board(sudoku.grid)}\n</input>\n")
    logger.print_and_log(f"<reasoning>")
    stack = []
    while True:
        
        # check state
        logger.print_and_log(f"<board>\n{format_board(sudoku.grid)}\n</board>")
        logger.print_and_log(f"<stack>\n{format_stack(stack)}\n</stack>")
        sudoku.update_possible_value_matrix()
        logger.print_and_log(f"=> Number of possibilities (estimate): {clean_possible_value_matrix(sudoku.possible_value_matrix)}")
        if sudoku.find_min_possible_value_position() is None:
            logger.print_and_log("[Sudoku is solved]")
            break
        else:
            logger.print_and_log("[Sudoku is not solved]")

        # fill number
        logger.print_and_log("<fill number>")
        row, col = sudoku.find_min_possible_value_position()
        logger.print_and_log(f"=> Minimum estimated value: ({row}, {col}) #{sudoku.possible_value_matrix[row][col]}, ")
        all_impossible_values = []
        logger.print_and_log("=> Impossible values in row: ", end="")
        for j, value in enumerate(sudoku.grid[row]):
            if value != 0:
                logger.print_and_log(f"{value} ", end="")
                all_impossible_values.append((row, j, value))
        logger.print_and_log("")
        logger.print_and_log("=> Impossible values in column: ", end="")
        for i in range(9):
            value = sudoku.grid[i][col]
            if value != 0:
                logger.print_and_log(f"{value} ", end="")
                all_impossible_values.append((i, col, value))
        logger.print_and_log("")
        logger.print_and_log("=> Impossible values in box: ", end="")
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                value = sudoku.grid[i][j]
                if value != 0:
                    logger.print_and_log(f"{value} ", end="")
                    all_impossible_values.append((i, j, value))
        logger.print_and_log("")

        all_impossible_values = sorted(list(set([value for _, _, value in all_impossible_values])))
        all_impossible_values_str = ' '.join(str(value) for value in all_impossible_values) + ' '
        logger.print_and_log(f"=> All impossible values: {all_impossible_values_str}")
        
        possible_values = sorted(list(set(range(1, 10)) - set(all_impossible_values)))
        possible_values_str = ' '.join(str(value) for value in possible_values) + ' ' if possible_values else 'None'
        logger.print_and_log(f"=> All possible values: {possible_values_str}")

        if not possible_values:
            # backtrack until possible value exists
            logger.print_and_log("[No possible value exists]")
            while True:
                last = stack.pop()
                row, col = last["cell"]
                possible_values = last["possible_values"]
                logger.print_and_log(f"=> Backtracking, pop from stack: ({row}, {col}) ")
                possible_values_str = ' '.join(str(value) for value in possible_values) + ' ' if possible_values else 'None'
                logger.print_and_log(f"=> Possible values: {possible_values_str}")
                if possible_values:
                    break
                else:
                    logger.print_and_log(f"[No possible value exists, reset cell]")
                    logger.print_and_log(f"> Fill cell ({row}, {col}) 0 ")
                    sudoku.grid[row][col] = 0
            
        logger.print_and_log("[Possible value exists]")
        num = possible_values[0]
        sudoku.grid[row][col] = num
        possible_values.remove(num)
        stack.append({"cell": (row, col), "possible_values": possible_values})

        logger.print_and_log(f"> Fill cell ({row}, {col}) {num} ")
        remaining_possible_values = ' '.join(str(value) for value in possible_values) + ' ' if possible_values else '- '
        logger.print_and_log(f"=> Remaining possible values: {remaining_possible_values}")
        logger.print_and_log(f"[update stack]")
        logger.print_and_log(f"<stack>\n{format_stack(stack)}\n</stack>")
        logger.print_and_log("</fill number>")
        
    logger.print_and_log('</reasoning>')
    logger.print_and_log(f"\n<output>\n{format_board(sudoku.grid)}\n</output>\n\n", end="")


def check_solution(solution, gt):
    for i in range(9):
        for j in range(9):
            if solution[i][j] != gt[i][j]:
                print(f"Error at cell ({i}, {j}): {solution[i][j]} != {gt[i][j]}")
                return False
    return True


def weighted_sample(ranges):
    rand = random.random()
    cumsum = 0
    
    for prob, (start, end) in ranges.items():
        cumsum += prob
        if rand <= cumsum:
            return random.randint(start, end - 1)
            
    last_range = list(ranges.values())[-1]
    return random.randint(last_range[0], last_range[1] - 1)


def generate_single_sudoku(difficulty_dict, seed=None):
    difficulty = weighted_sample(difficulty_dict)
    puzzle, gt = generate_sudoku(difficulty=difficulty, seed=seed)
    logger = Logger(False)
    sudoku = Sudoku(puzzle)
    solve_sudoku(sudoku, logger)
    solution = sudoku.grid
    if check_solution(solution, gt):
        return logger.log
    else:
        raise ValueError("Solution is incorrect")


def save_strings_to_jsonl(strings, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for string in strings:
            json_line = json.dumps({'text': string}, ensure_ascii=False)
            f.write(json_line + '\n')


def worker_function(worker_id, diff, base_seed):
    seed = base_seed + worker_id
    result = generate_single_sudoku(diff, seed=seed)
    return result

def stream_save_result(result, file_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        json_line = json.dumps({'text': result}, ensure_ascii=False)
        f.write(json_line + '\n')

def parallel_generate_sudoku(
    sample_count,
    difficulty,
    base_seed,
    output_file,
    num_processes=None
):
    if os.path.exists(output_file):
        os.remove(output_file)
    
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    pool = mp.Pool(processes=num_processes)
    
    worker = partial(
        worker_function,
        diff=difficulty,
        base_seed=base_seed
    )
    
    with tqdm(total=sample_count) as pbar:
        for i, result in enumerate(pool.imap_unordered(worker, range(sample_count))):
            stream_save_result(result, output_file)
            pbar.update(1)
    
    pool.close()
    pool.join()

if __name__ == '__main__':
    SAMPLE_COUNT = 10
    # Difficulty refers to the number of empty cells in the Sudoku puzzle.
    # Warning: Higher difficulties (more than 50 empty cells) will exponentially increase the time to generate the puzzle.
    DIFFICULTY = {
        0.3: (0, 42),
        0.65: (42, 53),
        0.05: (53, 60),
    }
    SEED = 0
    OUTPUT_FILE = 'sudoku_data.jsonl'
    
    parallel_generate_sudoku(
        sample_count=SAMPLE_COUNT,
        difficulty=DIFFICULTY,
        base_seed=SEED,
        output_file=OUTPUT_FILE
    )
