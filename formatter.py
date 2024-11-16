def format_board(board):
    formatted_rows = [' '.join(str(num) for num in inner_list) + ' ' for inner_list in board]
    return '\n'.join(formatted_rows)


def format_stack(stack):
    formatted_stack = ''
    for i, item in enumerate(stack):
        all_numbers_str = "".join(f'{str(num)} ' for num in item["possible_values"])
        all_numbers_str = '- ' if not all_numbers_str else all_numbers_str
        formatted_stack += f'{item["cell"]} {all_numbers_str}'
    return formatted_stack


def clean_possible_value_matrix(board):
    formatted_board = ""
    for i, row in enumerate(board):
        for j, num in enumerate(row):
            if num != 0:
                formatted_board += f"({i}, {j}) #{num}, "
    formatted_board += ""
    return formatted_board