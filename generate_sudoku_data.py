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
    logger.print_and_log(f"<input>\n{sudoku.grid}\n")
    logger.print_and_log(f"<reasoning>")
    stack = []
    while True:
        
        # check state
        logger.print_and_log("<check state>")
        logger.print_and_log(f"grid = {format_board(sudoku.grid)}")
        sudoku.update_possible_value_matrix()
        logger.print_and_log(f"possible values estmation = {clean_possible_value_matrix(sudoku.possible_value_matrix)}")
        if sudoku.find_min_possible_value_position() is None:
            logger.print_and_log("# sudoku is solved")
            logger.print_and_log('</reasoning>')
            break
        else:
            logger.print_and_log("# sudoku is not solved")
        logger.print_and_log(f"stack = {format_stack(stack)}")
        logger.print_and_log("</check state>")

        # fill number
        logger.print_and_log("<fill number>")
        row, col = sudoku.find_min_possible_value_position()
        logger.print_and_log("# find cell and possible values")
        logger.print_and_log(f"minimum estimated value = -{sudoku.possible_value_matrix[row][col]}-")
        logger.print_and_log(f"target cell = ({row}, {col}) ")
        all_impossible_values = []
        logger.print_and_log("impossible values in row:")
        for j, value in enumerate(sudoku.grid[row]):
            if value != 0:
                logger.print_and_log(f"({row}, {j}) {value}")
                all_impossible_values.append((row, j, value))
        logger.print_and_log("impossible values in column:")
        for i in range(9):
            value = sudoku.grid[i][col]
            if value != 0:
                logger.print_and_log(f"({i}, {col}) {value}")
                all_impossible_values.append((i, col, value))
        logger.print_and_log("impossible values in box:")
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                value = sudoku.grid[i][j]
                if value != 0:
                    logger.print_and_log(f"({i}, {j}) {value}")
                    all_impossible_values.append((i, j, value))

        logger.print_and_log(f"all impossible values:")
        for num in range(1, 10):
            logger.print_and_log(f"{num}: ", end="")
            no_impossible_value = True
            for cell in all_impossible_values:
                if cell[2] == num:
                    logger.print_and_log(f"{cell[0], cell[1]} ", end="")
                    no_impossible_value = False
            if no_impossible_value:
                logger.print_and_log("None", end="")
            logger.print_and_log("")
        possible_values = list(set(range(1, 10)) - set(value for _, _, value in all_impossible_values))
        logger.print_and_log(f"possible values = {possible_values}")

        if not possible_values:
            # backtrack until possible value exists
            logger.print_and_log("# no possible value exists")
            logger.print_and_log("# start backtracking")
            while True:
                logger.print_and_log("# pop from stack")
                last = stack.pop()
                row, col = last["cell"]
                possible_values = last["possible_values"]
                logger.print_and_log(f"target cell = ({row}, {col}) ")
                logger.print_and_log(f"possible values = {possible_values}")
                if possible_values:
                    break
                else:
                    logger.print_and_log("# no possible value exists")
                    logger.print_and_log("# reset cell")
                    logger.print_and_log(f"grid({row}, {col})  = 0")
                    sudoku.grid[row][col] = 0
            logger.print_and_log(f"# update stack")
            logger.print_and_log(f"stack = {format_stack(stack)}")
            
        logger.print_and_log("# possible value exists")
        num = possible_values[0]
        sudoku.grid[row][col] = num
        possible_values.remove(num)
        stack.append({"cell": (row, col), "possible_values": possible_values})

        logger.print_and_log(f"# fill cell")
        logger.print_and_log(f"grid({row}, {col})  = {num}")
        logger.print_and_log(f"# remaining possible values")
        logger.print_and_log(f"possible values = {possible_values}")
        logger.print_and_log(f"# update stack")
        logger.print_and_log(f"stack = {format_stack(stack)}")

        logger.print_and_log("</fill number>")
    logger.print_and_log(f"\n<output>\n{sudoku.grid}\n</output>\n\n", end="")


def check_solution(solution, gt):
    for i in range(9):
        for j in range(9):
            if solution[i][j] != gt[i][j]:
                print(f"Error at cell ({i}, {j}): {solution[i][j]} != {gt[i][j]}")
                return False
    return True


def generate_single_sudoku(minimum_difficulty, maximum_difficulty, seed=None):
    difficulty = random.randint(minimum_difficulty, maximum_difficulty)
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


def worker_function(worker_id, min_diff, max_diff, base_seed):
    seed = base_seed + worker_id
    result = generate_single_sudoku(min_diff, max_diff, seed=seed)
    return result

def stream_save_result(result, file_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        json_line = json.dumps({'text': result}, ensure_ascii=False)
        f.write(json_line + '\n')

def parallel_generate_sudoku(
    sample_count,
    min_difficulty,
    max_difficulty,
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
        min_diff=min_difficulty,
        max_diff=max_difficulty,
        base_seed=base_seed
    )
    
    with tqdm(total=sample_count) as pbar:
        for i, result in enumerate(pool.imap_unordered(worker, range(sample_count))):
            stream_save_result(result, output_file)
            pbar.update(1)
    
    pool.close()
    pool.join()

if __name__ == '__main__':
    SAMPLE_COUNT = 100
    # Difficulty refers to the number of empty cells in the Sudoku puzzle.
    # Warning: Higher difficulties (more than 50 empty cells) will exponentially increase the time to generate the puzzle.
    MINIMUM_DIFFICULTY = 40
    MAXIMUM_DIFFICULTY = 50
    SEED = 3
    OUTPUT_FILE = 'sudoku_data.jsonl'
    
    parallel_generate_sudoku(
        sample_count=SAMPLE_COUNT,
        min_difficulty=MINIMUM_DIFFICULTY,
        max_difficulty=MAXIMUM_DIFFICULTY,
        base_seed=SEED,
        output_file=OUTPUT_FILE
    )
