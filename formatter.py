def format_board(board):
    formatted_board = "["
    for i, row in enumerate(board):
        formatted_board += "["
        for j, num in enumerate(row):
            formatted_board += f"({i}, {j}) {num}"
            if j != 8:
                formatted_board += ", "
        formatted_board += "]"
        if i != 8:
            formatted_board += ", "
    formatted_board += "]"
    return formatted_board


def format_stack(stack):
    formatted_stack = "["
    for i, item in enumerate(stack):
        # formatted_stack += f'({item["cell"]} {item["possible_values"]})'
        all_numbers_str = "".join(str(num) for num in item["possible_values"])
        all_numbers_str = '-' if not all_numbers_str else all_numbers_str
        formatted_stack += f'{item["cell"]} {all_numbers_str}'
        if i != len(stack) - 1:
            formatted_stack += ", "
    formatted_stack += "]"
    return formatted_stack


def clean_possible_value_matrix(board):
    formatted_board = "["
    for i, row in enumerate(board):
        for j, num in enumerate(row):
            if num != 0:
                formatted_board += f"({i}, {j}) -{num}-"
                if i != 8 or j != 8:
                    formatted_board += ", "
    formatted_board += "]"
    return formatted_board