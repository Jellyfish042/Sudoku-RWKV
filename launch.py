import os

os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"

from rwkv_model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
from generate_sudoku_data import *
import json
import time
import copy
from utils import *


# class DummyPipeline:
#     def __init__(self):
#         self.data = self.read_jsonl("dummy_data.jsonl")

#     def generate(self, prompt, token_count, args, callback):
#         for s in self.data[0]["text"]:
#             callback(s)

#     def read_jsonl(self, file_path):
#         data = []
#         with open(file_path, "r", encoding="utf-8") as file:
#             for line in file:
#                 try:
#                     json_object = json.loads(line.strip())
#                     data.append(json_object)
#                 except json.JSONDecodeError as e:
#                     print(f"Error decoding JSON: {e}")
#         return data


class CoTLogger:
    def __init__(self, grid=None, verbose=False, real_time_verification=False):
        self.log = ""
        self.verbose = verbose
        self.real_time_verification = real_time_verification
        
        # for real-time verification
        self.sudoku = Sudoku(grid=grid)
        self.stack = []
        self.pointer = 0

    def print_and_log(self, text):
        if self.verbose:
            print(text, end="", flush=True)
        self.log += text
        
        if self.real_time_verification:
            
            found, position = self.search_string(self.log, "</fill number>", self.pointer)
            if found:
                new_step = self.log[self.pointer: position + len("</fill number>")]
                new_step = extract_between(new_step, '<check state>\n', '\n</fill number>')[0]
                passed, info = check_single_step(self.sudoku, self.stack, new_step, False)
                if not passed:
                    raise Exception(info)
                self.pointer = position + len("</fill number>")
            if len(self.log[self.pointer:]) > 10000:
                raise Exception("Exception Length")

    def clear_log(self):
        self.log = ""

    def get_token_count(self):
        return len(tokenizer.encode(self.log))
    
    def search_string(self, source: str, target: str, start_pos: int = 0) -> tuple[bool, int]:
        if not source or not target:
            return False, -1
        
        if start_pos < 0:
            start_pos = 0
        elif start_pos >= len(source):
            return False, -1
            
        if len(target) > len(source):
            return False, -1
        
        position = source.find(target, start_pos)
        
        if position == -1:
            return False, -1
        else:
            return True, position


def make_input(board):
    formatted_board = "<input>\n" + str(board)
    # formatted_board = "<input>\n" + str(board) + "\n\n<reasoning>"
    return formatted_board


def solve_sudoku_using_model(grid, verbose=False, max_token_count=500000, real_time_verification=False):
    begin = time.time()
    # print("Solving sudoku using RWKV...")
    logger = CoTLogger(grid=copy.deepcopy(grid), verbose=verbose)
    prompt = make_input(grid)
    is_completed = True
    
    if verbose:
        print(f'{" Model input ":-^100}')
        print(prompt)
        print(f'{" Model output ":-^100}')
        
    if real_time_verification:
        logger.real_time_verification = True
    try:
        pipeline.generate(prompt, max_token_count, gen_args, logger.print_and_log)
    except Exception as e:
        print(f' - Detected error during inference: {e}')
        is_completed = False
        
    token_count = len(tokenizer.encode(logger.log))
    if token_count == max_token_count:
        is_completed = False
        
    if verbose:
        print("-" * 100)
        print(f'Token count: {token_count}')
        print(f'Time: {time.time() - begin:.2f} seconds')
        if token_count == max_token_count:
            print("Token count reached the limit")
    
    return logger, is_completed


def compare_coordinates_and_calc_mae(str1, str2):
    def extract_values(s):
        parts = s.split('-')
        values = []
        format_check = []
        
        for i in range(len(parts)):
            if i % 2 == 1:
                try:
                    values.append(int(parts[i].strip()))
                    format_check.append('-')
                except ValueError:
                    return None, None
            else:
                format_check.append(parts[i])
                
        return values, ''.join(format_check)
    
    values1, format1 = extract_values(str1)
    values2, format2 = extract_values(str2)
    
    if values1 is None or values2 is None:
        return False, None
        
    if format1 != format2:
        return False, None
        
    if len(values1) != len(values2):
        return False, None
    
    absolute_errors = [abs(v1 - v2) for v1, v2 in zip(values1, values2)]
    mae = sum(absolute_errors) / len(absolute_errors)
    
    return True, mae


def find_min_value_coordinates(coord_str):
    coord_str = coord_str.strip()[1:-1].strip()
    
    if coord_str.endswith(']'):
        coord_str = coord_str[:-1]
        
    items = []
    current_item = ""
    depth = 0
    
    for char in coord_str:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            
        current_item += char
        
        if depth == 0 and char == '-' and current_item.count('-') == 2:
            items.append(current_item)
            current_item = ""
            
    if current_item.strip():
        items.append(current_item)
    
    min_value = float('inf')
    coords_dict = {}
    
    for item in items:
        item = item.strip().strip(',')
        if not item:
            continue
            
        parts = item.split('-')
        coord_str = parts[0].strip()  # (x, y)
        value = int(parts[1].strip())  # value
        
        if value < min_value:
            min_value = value
            coords_dict = {value: [coord_str]}
        elif value == min_value:
            if value not in coords_dict:
                coords_dict[value] = []
            coords_dict[value].append(coord_str)
    
    min_coords = [coord.strip().strip(',') for coord in coords_dict[min_value]]
    
    return min_coords, str(min_value)


def check_single_step(sudoku, stack, single_step_thought, verbose=True):
    
    sudoku.update_possible_value_matrix()
    
    lines = single_step_thought.split("\n")
    
    # check grid existence
    extracted_grid = lines.pop(0)
    if extracted_grid.startswith("grid = "):
        if verbose:
            print(f'{"grid found: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"grid found: ":<50}{"failed":>50}')
        print_detected_error(extracted_grid, "grid = " + format_board(sudoku.grid))
        return False, 'grid not found'
    # check grid value
    model_grid = extracted_grid[7:]
    gt_grid = format_board(sudoku.grid)
    if model_grid == gt_grid:
        if verbose:
            print(f'{"grid values: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"grid values: ":<50}{"failed":>50}')
        print_detected_error(model_grid, gt_grid)
        return False, 'grid values not matched'
    
    # check possible values estimation existence
    extracted_pve = lines.pop(0)
    if extracted_pve.startswith("possible values estmation = "):
        if verbose:
            print(f'{"possible values estimation found: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"possible values estimation found: ":<50}{"failed":>50}')
        print_detected_error(extracted_pve, "possible values estmation = " + clean_possible_value_matrix(sudoku.possible_value_matrix))
        return False, 'possible values estimation not found'
    # check possible values estimation
    model_pve = extracted_pve[28:]
    gt_pve = clean_possible_value_matrix(sudoku.possible_value_matrix)
    pve_passed, mae = compare_coordinates_and_calc_mae(model_pve, gt_pve)
    if pve_passed:
        if verbose:
            print(f'{"possible values estimation: ":<50}{f"passed (MAE = {round(mae, 3)})":>50}')
    else:
        if verbose:
            print(f'{"possible values estimation: ":<50}{"failed":>50}')
        print_detected_error(model_pve, gt_pve)
        return False, 'possible values estimation coordinates not matched'
    
    # check game over logic existence
    extracted_game_over = lines.pop(0)
    if extracted_game_over.startswith("# sudoku is "):
        if verbose:
            print(f'{"game over logic found: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"game over logic found: ":<50}{"failed":>50}')
        print_detected_error(extracted_game_over, "# sudoku is solved" if gt_pve == "[]" else "# sudoku is not solved")
        return False, 'game over logic not found'
    # check game over logic
    model_game_over = extracted_game_over[12:]
    possible_values = clean_possible_value_matrix(sudoku.possible_value_matrix)
    if possible_values == "[]":
        gt_game_over = "solved"
    else:
        gt_game_over = "not solved"
    if model_game_over == gt_game_over:
        if verbose:
            print(f'{"game over logic: ":<50}{"passed":>50}')
        if model_game_over == "solved":
            return True, 'solved'
    else:
        if verbose:
            print(f'{"game over logic: ":<50}{"failed":>50}')
        print_detected_error(model_game_over, gt_game_over)
        return False, 'game over logic not matched'
    
    # check stack existence
    extracted_stack = lines.pop(0)
    if extracted_stack.startswith("stack = "):
        if verbose:
            print(f'{"stack found: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"stack found: ":<50}{"failed":>50}')
        print_detected_error(extracted_stack, "stack = " + format_stack(stack))
        return False, 'stack not found'
    # check stack values
    model_stack = extracted_stack[8:]
    gt_stack = format_stack(stack)
    if model_stack == gt_stack:
        if verbose:
            print(f'{"stack values: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"stack values: ":<50}{"failed":>50}')
        print_detected_error(model_stack, gt_stack)
        return False, 'stack values not matched'
    
    # check intermediate comment token
    fill_number_token_1 = lines.pop(0)
    fill_number_token_2 = lines.pop(0)
    fill_number_token_3 = lines.pop(0)
    if fill_number_token_1 == '</check state>' and fill_number_token_2 == '<fill number>' and fill_number_token_3 == '# find cell and possible values':
        if verbose:
            print(f'{"intermediate comment token: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"intermediate comment token: ":<50}{"failed":>50}')
        print_detected_error(fill_number_token_1 + '\n' + fill_number_token_2 + '\n' + fill_number_token_3, "</check state>\n<fill number>\n# find cell and possible values")
        return False, 'intermediate comment token not found'
    
    # check minimum estimated value existence
    minimum_estimated_value = lines.pop(0)
    if minimum_estimated_value.startswith("minimum estimated value = -"):
        if verbose:
            print(f'{"minimum estimated value found: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"minimum estimated value found: ":<50}{"failed":>50}')
        print_detected_error(minimum_estimated_value, "minimum estimated value = -x-")
        return False, 'minimum estimated value not found'
    # check minimum estimated value
    model_estimated_value = minimum_estimated_value[27:28]
    gt_min_coords, gt_min_value = find_min_value_coordinates(model_pve)
    if model_estimated_value == gt_min_value:
        if verbose:
            print(f'{"minimum estimated value: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"minimum estimated value: ":<50}{"failed":>50}')
        print_detected_error(minimum_estimated_value, f"minimum estimated value = -{gt_min_value}-")
        return False, 'minimum estimated value not matched'
    
    # check min value coordinates existence
    model_target_cell = lines.pop(0)
    if model_target_cell.startswith("target cell = "):
        if verbose:
            print(f'{"min value coordinates found: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"min value coordinates found: ":<50}{"failed":>50}')
        print_detected_error(model_target_cell, "target cell = (x, y)")
        return False, 'min value coordinates not found'
    # check min value coordinates
    model_target_cell = model_target_cell[14:20]
    if model_target_cell in gt_min_coords:
        if verbose:
            print(f'{"min value coordinates: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"min value coordinates: ":<50}{"failed":>50}')
        print_detected_error(model_target_cell, f'one of {gt_min_coords}')
        return False, 'min value coordinates not matched'
    
    # check impossible value in row, column, box existence
    row_token_exist, col_token_exist, box_token_exist = False, False, False
    all_impossible_value_token_exist = False
    model_predict_str = ''
    for _ in range(8 * 3 + 3 + 1):
        new_line = lines.pop(0)
        if new_line == "impossible values in row:":
            row_token_exist = True
        elif new_line == "impossible values in column:":
            col_token_exist = True
        elif new_line == "impossible values in box:":
            box_token_exist = True
        if new_line == 'all impossible values:':
            all_impossible_value_token_exist = True
            break
        model_predict_str += new_line + '\n'
    model_predict_str = model_predict_str.strip()
    if row_token_exist and col_token_exist and box_token_exist:
        if verbose:
            print(f'{"impossible value in row, column, box found: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"impossible value in row, column, box found: ":<50}{"failed":>50}')
        print_detected_error(model_predict_str, "impossible values in row:\n(x, y) z\nimpossible values in column:\n(x, y) z\nimpossible values in box:\n(x, y) z")
        return False, 'impossible value in row, column, box not found'
    
    # check impossible value in row, column, box
    row = int(model_target_cell[1:2])
    col = int(model_target_cell[4:5])
    
    all_impossible_values = []
    
    gt_impossible_values = ''
    gt_impossible_values += "impossible values in row:\n"
    for j, value in enumerate(sudoku.grid[row]):
        if value != 0:
            gt_impossible_values += f"({row}, {j}) {value}\n"
            all_impossible_values.append((row, j, value))
    gt_impossible_values += "impossible values in column:\n"
    for i in range(9):
        value = sudoku.grid[i][col]
        if value != 0:
            gt_impossible_values += f"({i}, {col}) {value}\n"
            all_impossible_values.append((i, col, value))
    gt_impossible_values += "impossible values in box:\n"
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(box_row, box_row + 3):
        for j in range(box_col, box_col + 3):
            value = sudoku.grid[i][j]
            if value != 0:
                gt_impossible_values += f"({i}, {j}) {value}\n"
                all_impossible_values.append((i, j, value))
    gt_impossible_values = gt_impossible_values.strip()
    if model_predict_str == gt_impossible_values:
        if verbose:
            print(f'{"impossible value in row, column, box: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"impossible value in row, column, box: ":<50}{"failed":>50}')
        print_detected_error(model_predict_str, gt_impossible_values)
        return False, 'impossible value in row, column, box not matched'
    
    # check all impossible values existence
    if all_impossible_value_token_exist:
        if verbose:
            print(f'{"all impossible values found: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"all impossible values found: ":<50}{"failed":>50}')
        print_detected_error(lines[0], "all impossible values:")
        return False, 'all impossible values not found'
    # check all impossible values
    gt_all_impossible_value = ''
    for num in range(1, 10):
        gt_all_impossible_value += f"{num}: "
        no_impossible_value = True
        for cell in all_impossible_values:
            if cell[2] == num:
                gt_all_impossible_value += f"{cell[0], cell[1]} "
                no_impossible_value = False
        if no_impossible_value:
            gt_all_impossible_value += "None"
        gt_all_impossible_value += '\n'
    model_predict_all_impossible_value = ''
    for i in range(9):
        new_line = lines.pop(0)
        model_predict_all_impossible_value += new_line + '\n'
    if model_predict_all_impossible_value == gt_all_impossible_value:
        if verbose:
            print(f'{"all impossible values: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"all impossible values: ":<50}{"failed":>50}')
        print_detected_error(model_predict_all_impossible_value, gt_all_impossible_value)
        return False, 'all impossible values not matched'
        
    # check possible values existence
    extracted_possible_values = lines.pop(0)
    if extracted_possible_values.startswith("possible values = "):
        if verbose:
            print(f'{"possible values found: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"possible values found: ":<50}{"failed":>50}')
        print_detected_error(extracted_possible_values, "possible values = [xxx, xxx, ...]")
        return False, 'possible values not found'
    # check possible values
    model_pred_possible_values = extracted_possible_values[18:]
    gt_possible_values = list(set(range(1, 10)) - set(value for _, _, value in all_impossible_values))
    if model_pred_possible_values == str(gt_possible_values):
        if verbose:
            print(f'{"possible values: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"possible values: ":<50}{"failed":>50}')
        print_detected_error(model_pred_possible_values, str(gt_possible_values))
        return False, 'possible values not matched'
    
    # check possible value exist logic existence
    model_possible_value_exist = lines.pop(0)
    if model_possible_value_exist.startswith("# possible value exists") or model_possible_value_exist.startswith("# no possible value exists"):
        if verbose:
            print(f'{"possible value exist logic found: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"possible value exist logic found: ":<50}{"failed":>50}')
        print_detected_error(model_possible_value_exist, "# possible value exists or # no possible value exists")
        return False, 'possible value exist logic not found'
    # check possible value exist logic
    gt_possible_value_exist = "# possible value exists" if gt_possible_values else "# no possible value exists"
    if model_possible_value_exist == gt_possible_value_exist:
        if verbose:
            print(f'{"possible value exist logic: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"possible value exist logic: ":<50}{"failed":>50}')
        print_detected_error(model_possible_value_exist, gt_possible_value_exist)
        return False, 'possible value exist logic not matched'
    
    # check decision token
    mp_decision_token = lines.pop(0)
    gt_decision_token = '# fill cell' if gt_possible_value_exist == "# possible value exists" else '# start backtracking'
    if mp_decision_token == gt_decision_token:
        if verbose:
            print(f'{"decision token: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"decision token: ":<50}{"failed":>50}')
        print_detected_error(mp_decision_token, gt_decision_token)
        return False, 'decision token not found'
    
    if gt_decision_token == '# start backtracking':  # check backtracking logic
        idx = 1
        while True:
            # check pop stack logic existence
            mp_stack_pre_token = lines.pop(0)
            mp_pop_cell = lines.pop(0)
            mp_pop_values = lines.pop(0)
            mp_is_possible_value_exist = lines.pop(0)
            if mp_stack_pre_token == '# pop from stack' \
                and mp_pop_cell.startswith('target cell = (') \
                    and mp_pop_values.startswith('possible values = ') \
                        and mp_is_possible_value_exist in ['# update stack', '# no possible value exists']:
                if verbose:
                    print(f'{f"pop stack logic found ({idx})":<50}{"passed":>50}')
            else:
                if verbose:
                    print(f'{f"pop stack logic found ({idx})":<50}{"failed":>50}')
                print_detected_error(mp_stack_pre_token + '\n' + mp_pop_cell + '\n' + mp_pop_values + '\n' + mp_is_possible_value_exist, 
                                     "# pop from stack\ntarget cell = (x, y)\npossible values = [xxx, xxx, ...]\n# update stack or # no possible value exists")
                return False, 'pop stack logic not found'
            # check the popped value
            last = stack.pop()
            row, col = last["cell"]
            gt_possible_values = last["possible_values"]
            gt_pop_cell = f'target cell = ({row}, {col}) '
            gt_pop_values = f'possible values = {gt_possible_values}'
            if mp_pop_cell == gt_pop_cell and mp_pop_values == gt_pop_values:
                if verbose:
                    print(f'{f"popped value ({idx})":<50}{"passed":>50}')
            else:
                if verbose:
                    print(f'{f"popped value ({idx})":<50}{"failed":>50}')
                print_detected_error(mp_pop_cell + '\n' + mp_pop_values, gt_pop_cell + '\n' + gt_pop_values)
                return False, 'popped value not matched'
            # check model ckeck value
            gt_check_value_token = '# update stack' if gt_possible_values else '# no possible value exists'
            if mp_is_possible_value_exist == gt_check_value_token:
                if verbose:
                    print(f'{f"ckeck value token ({idx})":<50}{"passed":>50}')
            else:
                if verbose:
                    print(f'{f"ckeck value token ({idx})":<50}{"failed":>50}')
                print_detected_error(mp_is_possible_value_exist, gt_check_value_token)
                return False, 'ckeck value token not matched'
            
            if gt_check_value_token == '# update stack':  # end of backtracking
                # check new stack existence
                mp_new_stack = lines.pop(0)
                if mp_new_stack.startswith('stack = '):
                    if verbose:
                        print(f'{f"new stack found ({idx})":<50}{"passed":>50}')
                else:
                    if verbose:
                        print(f'{f"new stack found ({idx})":<50}{"failed":>50}')
                    print_detected_error(mp_new_stack, f'stack = {format_stack(stack)}')
                    return False, 'new stack not found'
                # check new stack
                gt_new_stack = format_stack(stack)
                if mp_new_stack[8:] == gt_new_stack:
                    if verbose:
                        print(f'{f"new stack ({idx})":<50}{"passed":>50}')
                else:
                    if verbose:
                        print(f'{f"new stack ({idx})":<50}{"failed":>50}')
                    print_detected_error(mp_new_stack, f'stack = {format_stack(stack)}')
                    return False, 'new stack not matched'
                lines.pop(0)
                lines.pop(0)
                break
            
            # check reset cell logic existence
            mp_reset_pre_token = lines.pop(0)
            mp_reset = lines.pop(0)
            if mp_reset_pre_token == '# reset cell' and mp_reset.startswith('grid('):
                if verbose:
                    print(f'{f"reset cell logic found ({idx})":<50}{"passed":>50}')
            else:
                if verbose:
                    print(f'{f"reset cell logic found ({idx})":<50}{"failed":>50}')
                print_detected_error(mp_reset_pre_token + '\n' + mp_reset, "# reset cell\ngrid(x, y) = 0")
                return False, 'reset cell logic not found'
            # check reset cell logic
            sudoku.grid[row][col] = 0  # real reset
            gt_reset = f'grid({row}, {col})  = 0'
            if mp_reset == gt_reset:
                if verbose:
                    print(f'{f"reset cell logic ({idx})":<50}{"passed":>50}')
            else:
                if verbose:
                    print(f'{f"reset cell logic ({idx})":<50}{"failed":>50}')
                print_detected_error(mp_reset, gt_reset)
                return False, 'reset cell logic not matched'
            
            idx += 1
    
    # fill cell logic
    extracted_fill_cell = lines.pop(0)
    if extracted_fill_cell.startswith('grid('):
        if verbose:
            print(f'{"fill cell logic found: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"fill cell logic found: ":<50}{"failed":>50}')
        print_detected_error(extracted_fill_cell, f'grid({row}, {col})  = {gt_possible_values[0]}')
        return False, 'fill cell logic not found'
    # check fill cell logic
    sudoku.grid[row][col] = gt_possible_values[0]  # fill the cell
    gt_fill_cell = f'grid({row}, {col})  = {gt_possible_values[0]}'
    if extracted_fill_cell == gt_fill_cell:
        if verbose:
            print(f'{"fill cell logic: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"fill cell logic: ":<50}{"failed":>50}')
        print_detected_error(extracted_fill_cell, gt_fill_cell)
        return False, 'fill cell logic not matched'
    # check remaining possible values existence
    remaining_possible_values_pre_token = lines.pop(0)
    remaining_possible_values = lines.pop(0)
    if remaining_possible_values_pre_token == '# remaining possible values' and remaining_possible_values.startswith('possible values = '):
        if verbose:
            print(f'{"remaining possible values found: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"remaining possible values found: ":<50}{"failed":>50}')
        print_detected_error(remaining_possible_values_pre_token + '\n' + remaining_possible_values, 
                             f'# remaining possible values\npossible values = {gt_possible_values[1:]}')
        return False, 'remaining possible values not found'
    # check remaining possible values
    mp_remaining_possible_values = remaining_possible_values[18:]
    gt_possible_values.pop(0)
    gt_remaining_possible_values = gt_possible_values
    if str(gt_remaining_possible_values) == mp_remaining_possible_values:
        if verbose:
            print(f'{"remaining possible values: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"remaining possible values: ":<50}{"failed":>50}')
        print_detected_error(remaining_possible_values, f'possible values = {gt_remaining_possible_values}')
        return False, 'remaining possible values not matched'
    # check update stack logic existence
    mp_update_stack_pre_token = lines.pop(0)
    mp_update_stack = lines.pop(0)
    if mp_update_stack_pre_token == '# update stack' and mp_update_stack.startswith('stack = '):
        if verbose:
            print(f'{"update stack logic found: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"update stack logic found: ":<50}{"failed":>50}')
        print_detected_error(mp_update_stack, f'stack = [xxx, xxx, ...]')
        return False, 'update stack logic not found'
    # check update stack logic
    mp_new_stack = mp_update_stack[8:]
    stack.append({"cell": (row, col), "possible_values": gt_remaining_possible_values})
    gt_new_stack = format_stack(stack)
    if mp_new_stack == gt_new_stack:
        if verbose:
            print(f'{"update stack logic: ":<50}{"passed":>50}')
    else:
        if verbose:
            print(f'{"update stack logic: ":<50}{"failed":>50}')
        print_detected_error(mp_new_stack, gt_new_stack)
        return False, 'updated stack not matched'

    return True, 'all passed'


def parse_output_content(text):
    start_tag = "<output>"
    end_tag = "</output>"
    start_index = text.find(start_tag) + len(start_tag)
    end_index = text.find(end_tag)
    
    if start_index == -1 or end_index == -1:
        return False, None
        
    content = text[start_index:end_index].strip()
    
    try:
        content = content.strip('[]')
        rows = content.split('],')
        
        result = []
        for row in rows:
            # Clean up the row string
            row = row.strip(' []')
            if not row:
                continue
                
            # Split by comma and convert to integers
            numbers = [int(num.strip()) for num in row.split(',')]
            result.append(numbers)
            
        return True, result
    except ValueError:
        return False, None
    

def compare_2d_lists(list1, list2):
    if not isinstance(list1, list) or not isinstance(list2, list):
        return False
        
    if len(list1) != len(list2):
        return False
        
    for row1, row2 in zip(list1, list2):
        if not isinstance(row1, list) or not isinstance(row2, list):
            return False
            
        if len(row1) != len(row2):
            return False
            
        if not all(isinstance(x, int) and isinstance(y, int) and x == y 
                  for x, y in zip(row1, row2)):
            return False
            
    return True


def check_output(output, gt, verbose=True):
    mp_answer_exist, mp_answer = parse_output_content(output + '</output>')
    if not mp_answer_exist:
        if verbose:
            print(f'{"output found: ":<50}{"failed":>50}')
            print(f' - Model Prediction: {output}')
            print(f' - Ground Truth: {gt}')
        return False
    if verbose:
        print(f'{"output found: ":<50}{"passed":>50}')
    # print(gt, mp_answer)
    success = compare_2d_lists(gt, mp_answer)
    if success:
        if verbose:
            print(f'{"output values: ":<50}{"passed":>50}')
            print_sudoku_comparison(mp_answer, gt, 'Model Prediction', 'Ground Truth')
    else:
        if verbose:
            print(f'{"output values: ":<50}{"failed":>50}')
            print_sudoku_comparison(mp_answer, gt, 'Model Prediction', 'Ground Truth')
    return success

    
def check_cot(input_grid, gt, cot, verify_intermediate_step=False, verbose=False):
    sudoku = Sudoku(grid=input_grid)
    stack = []
    split_cot = cot.split("<check state>")
    split_cot = [x.strip() for x in split_cot]
    split_cot.pop(0)
    output = split_cot.pop(-1)
    
    # check intermediate output
    if verify_intermediate_step:
        for i, single_step_thought in enumerate(split_cot):
            if verbose:
                print(f'{f" Verifying Step {i+1} ":=^100}')
            passed, _ = check_single_step(sudoku, stack, single_step_thought, verbose=verbose)
            if not passed:
                return False
        
    # check final output
    if verbose:
        print(f'{" Verifying output ":=^100}')
    success = check_output(output, gt, verbose=verbose)
    if not success:
        return False
    else:
        return True


MODEL_PATH = "sudoku_rwkv_20241029.pth"
print(f"Loading model {MODEL_PATH}...")
model = RWKV(model=MODEL_PATH, strategy="cuda fp16", verbose=False)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
tokenizer = TRIE_TOKENIZER("sudoku_vocab_v6.txt")
pipeline.tokenizer = TRIE_TOKENIZER("sudoku_vocab_v6.txt")
gen_args = PIPELINE_ARGS(
    top_k=1,
    alpha_frequency=0,
    alpha_presence=0,
    token_stop=[127],
)

# only for testing
# pipeline = DummyPipeline()

HELP = """
Available Commands:
    i     - Input your own sudoku puzzle
    n     - Generate a new sudoku puzzle
    s     - Solve current sudoku using RWKV
    d     - Change difficulty level
    r     - Set random seed for sudoku generation
    v     - Toggle real-time verification
    e     - Evaluate model on N different samples
    q     - Quit program
"""
DIFFICULTY_NOTE = '''Difficulty refers to the number of empty cells in the Sudoku puzzle.
Warning: Higher difficulties (more than 50 empty cells) will exponentially increase the token usage for solving. Use with caution.'''

RTV_NOTE = '''
Notes:
 • When enabled, model inference will stop if any error is detected
 • Note: Detected errors don't always mean the Sudoku can't be solved
 • Sometimes these are only temporary or non-critical errors
 '''
 
RANDOM_SEED_NOTE = '''When a random seed is set, the same Sudoku puzzle will be generated each time with the same seed.
Otherwise, a new seed will be generated each time, resulting in a different Sudoku puzzle.'''

# testing parameters
# default_difficulty = 50
# base_seed = 3

default_difficulty = 40
base_seed = None
default_max_token_count = 500000
break_when_something_wrong = True  # real-time verification

current_seed = base_seed if base_seed is not None else int(time.time())
grid, solved_grid = generate_sudoku(difficulty=default_difficulty, seed=current_seed)
grid, solved_grid = None, None

while True:
    
    print('-' * 100)
    print(f'DIFFICULTY: {default_difficulty} | SEED: {current_seed} | MAX TOKEN COUNT: {default_max_token_count} | BREAK WHEN SOMETHING WRONG: {break_when_something_wrong}')
    print("Sudoku:")
    if grid is not None:
        print_sudoku(grid)
    else:
        print("None")
    print(HELP)
        
    op = input("Enter command: ")
    if op == "q":
        break
    elif op == "d":
        print(DIFFICULTY_NOTE)
        default_difficulty = int(input("Enter difficulty: "))
        grid, solved_grid = generate_sudoku(difficulty=default_difficulty, seed=current_seed)
    elif op == "n":
        grid, solved_grid = generate_sudoku(difficulty=default_difficulty, seed=current_seed)
    elif op == "i":
        while True:
            temp_grid = [[0] * 9 for _ in range(9)]
            print("Enter the sudoku grid (0 for empty cells, space separated):")
            for i in range(9):
                row = input(f"Enter row {i + 1}: ")
                numbers = [int(x) for x in row.split()]
                for j in range(9):
                    temp_grid[i][j] = numbers[j]
            solution_count, solved_grid_temp = solve_sudoku_gt(temp_grid)
            if solution_count == 0:
                print("Invalid sudoku: no solution")
                temp_grid = None
            elif solution_count > 1:
                print("Invalid sudoku: multiple solutions")
                temp_grid = None
            else:
                print("Sudoku input successfully!")
                grid = temp_grid
                solved_grid = solved_grid_temp
                break 
    elif op == "s":
        if grid is None:
            print("No sudoku generated yet!")
        else:
            print("Solving sudoku using RWKV...")
            logger, is_completed = solve_sudoku_using_model(copy.deepcopy(grid), 
                                                            verbose=True, 
                                                            max_token_count=default_max_token_count, 
                                                            real_time_verification=break_when_something_wrong)
            if is_completed:
                cot = logger.log
                success = check_cot(grid, solved_grid, cot, verify_intermediate_step=False, verbose=True)
    elif op == "e":
        n = int(input("Enter number of samples: "))
        all_samples = []
        print(f'SETTINGS: DIFFICULTY: {default_difficulty} | SEED: {current_seed} | MAX TOKEN COUNT: {default_max_token_count} | SAMPLES: {n}')
        print("Generating samples...")
        for i in tqdm(range(n)):
            grid_i, solved_grid_i = generate_sudoku(difficulty=default_difficulty, seed=current_seed + i)
            all_samples.append((grid_i, solved_grid_i))
        print("Solving samples...")
        result_correct_count = 0
        perfectly_correct_count = 0
        token_usage = 0
        for grid_i, solved_grid_i in tqdm(all_samples):
            logger, is_completed = solve_sudoku_using_model(copy.deepcopy(grid_i), 
                                                            verbose=False, 
                                                            max_token_count=default_max_token_count, 
                                                            real_time_verification=break_when_something_wrong)
            cot = logger.log
            token_usage += len(tokenizer.encode(cot))
            result_correct = check_cot(grid_i, solved_grid_i, cot, verify_intermediate_step=False, verbose=False)
            perfectly_correct = check_cot(grid_i, solved_grid_i, cot, verify_intermediate_step=True, verbose=False)
            if result_correct:
                result_correct_count += 1
            if perfectly_correct:
                perfectly_correct_count += 1
        print(f"Result correct samples: {result_correct_count}/{n}  Accuracy: {result_correct_count / n:.3f}")
        print(f"Perfectly correct samples: {perfectly_correct_count}/{n}  Accuracy: {perfectly_correct_count / n:.3f}")
        print(f"Average token usage: {token_usage / n:.2f}")
    elif op == "r":
        print(RANDOM_SEED_NOTE)
        base_seed = int(input("Enter seed: "))
    elif op == "v":
        break_when_something_wrong = not break_when_something_wrong
        print(f"Real-time verification: {break_when_something_wrong}")
        print(RTV_NOTE)
    else:
        print("Invalid command")
            
    current_seed = base_seed if base_seed is not None else int(time.time())
