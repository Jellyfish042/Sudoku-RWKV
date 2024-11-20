import tkinter as tk
from tkinter import messagebox, font
import random
import time
from threading import Thread
import copy
import os
from formatter import *
from utils import solve_sudoku_gt
from generate_sudoku_data import generate_sudoku as gen


SEED = None
DEFAULT_DIFFICULTY = 55
MODEL_PATH = 'sudoku_rwkv_20241120.pth'


class ModernSudokuGame:
    def __init__(self, master):
        self.master = master
        self.master.title("Sudoku")
        # self.master.geometry("1100x800")
        self.master.geometry("1100x750")
        self.master.resizable(False, False)

        self.colors = {
            "bg": "#1a1b26",
            "tile": "#7aa2f7",
            "fixed": "#9ece6a",
            "empty": "#414868",
            "text": "#a9b1d6",
            "button_bg": "#414868",
            "button_hover": "#565f89",
            "tile_text": "#ffffff",
            "error": "#f7768e",
            "selected": "#bb9af7",
            "glow_unsolved": "#f7768e",
            "glow_solved": "#9ece6a"
        }

        self.show_conflicts = False

        self.model = RWKVModel()
        self.initial_state = None
        self.current_state = None
        self.fixed_numbers = set()
        self.ai_filled_numbers = set()
        self.selected_cell = None
        self.time = 0
        self.timer_running = False
        self.timer_id = None
        self.is_model_running = False
        self.manual_control = True
        self.total_tokens = 0
        
        self.buttons = [[None for _ in range(9)] for _ in range(9)]
        self.create_layout()
        self.new_game()

    def create_layout(self):
        self.main_layout = tk.Frame(self.master, bg=self.colors["bg"])
        self.main_layout.pack(expand=True, fill="both")

        self.create_reasoning_display()

        self.game_container = tk.Frame(self.main_layout, bg=self.colors["bg"])
        self.game_container.pack(side="right", expand=True, fill="both", padx=20, pady=20)

        self.create_title()
        self.create_stats_display()
        self.create_instruction()
        self.create_game_grid()
        self.create_control_buttons()
        
    def create_instruction(self):
        instruction = tk.Label(
            self.game_container,
            text="Click a cell and type 1-9 to fill, 0 to clear",
            font=font.Font(family="Helvetica", size=12),
            bg=self.colors["bg"],
            fg=self.colors["text"]
        )
        instruction.pack(pady=(0, 10))

    def create_reasoning_display(self):
        self.reasoning_container = tk.Frame(
            self.main_layout, bg=self.colors["bg"], width=450)
        self.reasoning_container.pack(side="left", fill="both", padx=20, pady=20)
        self.reasoning_container.pack_propagate(False)

        title = tk.Label(self.reasoning_container, text="Reasoning Process",
                        font=font.Font(family="Helvetica", size=16, weight="bold"),
                        bg=self.colors["bg"], fg=self.colors["text"])
        title.pack(pady=(0, 10))

        self.token_frame = tk.Frame(self.reasoning_container, bg=self.colors["bg"])
        self.token_frame.pack(fill="x", pady=(0, 10))

        tk.Label(self.token_frame, text="Tokens Used",
                font=font.Font(family="Helvetica", size=12),
                bg=self.colors["bg"], fg=self.colors["text"]).pack(side="left")

        self.token_count = tk.Label(self.token_frame, text="0",
                                  font=font.Font(family="Helvetica", size=12),
                                  bg=self.colors["bg"], fg=self.colors["text"])
        self.token_count.pack(side="right")

        text_frame = tk.Frame(self.reasoning_container, bg=self.colors["bg"])
        text_frame.pack(expand=True, fill="both")

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")

        self.reasoning_text = tk.Text(text_frame, wrap=tk.WORD,
                                    bg=self.colors["button_bg"],
                                    fg=self.colors["tile_text"],
                                    font=("Consolas", 12),
                                    padx=10, pady=10, maxundo=0)
        
        self.reasoning_text.pack(expand=True, fill="both")
        self.reasoning_text.tag_configure("move", foreground="#9ece6a")
        
        self.reasoning_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.configure(command=self.reasoning_text.yview)

    def create_title(self):
        title = tk.Label(self.game_container, text="SUDOKU",
                        font=font.Font(family="Helvetica", size=24, weight="bold"),
                        bg=self.colors["bg"], fg=self.colors["text"])
        title.pack(pady=(0, 20))

    def create_stats_display(self):
        # info_frame = tk.Frame(self.game_container, bg=self.colors["bg"])
        # info_frame.pack(fill="x", pady=(0, 20))

        # time_frame = tk.Frame(info_frame, bg=self.colors["bg"])
        # time_frame.pack(expand=True)

        # tk.Label(time_frame, text="TIME",
        #         font=font.Font(family="Helvetica", size=12),
        #         bg=self.colors["bg"], fg=self.colors["text"]).pack()

        # self.time_label = tk.Label(time_frame, text="0s",
        #                          font=font.Font(family="Helvetica", size=20, weight="bold"),
        #                          bg=self.colors["bg"], fg=self.colors["text"])
        # self.time_label.pack()
        
        info_frame = tk.Frame(self.game_container, bg=self.colors["bg"])
        time_frame = tk.Frame(info_frame, bg=self.colors["bg"])
        self.time_label = tk.Label(time_frame, text="0s")

    def create_game_grid(self):
        outer_padding = 15
        glow_thickness = 4
        
        self.outer_frame = tk.Frame(
            self.game_container,
            bg=self.colors["glow_unsolved"],
            padx=glow_thickness,
            pady=glow_thickness
        )
        self.outer_frame.pack(padx=10, pady=10)

        middle_frame = tk.Frame(
            self.outer_frame,
            bg=self.colors["bg"],
            padx=outer_padding,
            pady=outer_padding
        )
        middle_frame.pack(fill="both", expand=True)

        self.game_frame = tk.Frame(middle_frame, bg=self.colors["bg"])
        self.game_frame.pack()

        button_size = 45
        for i in range(9):
            for j in range(9):
                frame = tk.Frame(
                    self.game_frame,
                    width=button_size,
                    height=button_size,
                    bg=self.colors["bg"]
                )
                padx = (5 if j % 3 == 0 else 1, 5 if j % 3 == 2 else 1)
                pady = (5 if i % 3 == 0 else 1, 5 if i % 3 == 2 else 1)
                frame.grid(row=i, column=j, padx=padx, pady=pady)
                frame.grid_propagate(False)

                button = tk.Button(
                    frame,
                    font=font.Font(family="Helvetica", size=16, weight="bold"),
                    borderwidth=0,
                    command=lambda row=i, col=j: self.cell_click(row, col)
                )
                button.place(relwidth=1, relheight=1)

                button.bind("<Enter>", lambda e, btn=button: self.on_hover(btn))
                button.bind("<Leave>", lambda e, btn=button: self.on_leave(btn))

                self.buttons[i][j] = button

            self.game_frame.bind("<Key>", self.on_key_press)
            self.game_frame.focus_set()
                
    def on_double_click(self, row, col):
        """Handle direct number input on double click"""
        if not self.manual_control:
            return
            
        if (row, col) in self.fixed_numbers:
            return

        # Create a simple entry dialog
        dialog = tk.Toplevel(self.master)
        dialog.title("Enter Number")
        dialog.geometry("200x100")
        dialog.resizable(False, False)
        dialog.configure(bg=self.colors["bg"])
        dialog.transient(self.master)
        dialog.grab_set()

        # Create and configure the entry widget
        entry = tk.Entry(dialog, font=("Helvetica", 16), justify="center", width=10)
        entry.pack(pady=20)
        entry.focus_set()

        def validate_and_set(event=None):
            value = entry.get().strip()
            try:
                num = int(value)
                if num in range(0, 10):  # Accept 0-9
                    self.cell_click(row, col)  # Select the cell
                    self.number_click(num)  # Set the number (0 will clear the cell)
                    dialog.destroy()
                else:
                    messagebox.showwarning("Invalid Input", "Please enter a number between 0 and 9")
                    entry.delete(0, tk.END)
                    entry.focus_set()
            except ValueError:
                messagebox.showwarning("Invalid Input", "Please enter a valid number")
                entry.delete(0, tk.END)
                entry.focus_set()

        entry.bind('<Return>', validate_and_set)
        
        ok_button = tk.Button(
            dialog,
            text="OK",
            command=validate_and_set,
            bg=self.colors["button_bg"],
            fg=self.colors["tile_text"]
        )
        ok_button.pack(pady=5)

    def create_control_buttons(self):
        self.control_frame = tk.Frame(self.game_container, bg=self.colors["bg"])
        self.control_frame.pack(pady=20)

        buttons_info = [
            ("New Game", self.new_game),
            ("Start Model", self.start_model)
        ]

        for text, command in buttons_info:
            btn = tk.Button(
                self.control_frame,
                text=text,
                font=font.Font(family="Helvetica", size=12),
                command=command,
                bg=self.colors["button_bg"],
                fg=self.colors["tile_text"],
                borderwidth=0,
                padx=15,
                pady=8
            )
            btn.pack(side="left", padx=5)

            btn.bind("<Enter>", lambda e, b=btn: b.configure(
                bg=self.colors["button_hover"]) if b["state"] != "disabled" else None)
            btn.bind("<Leave>", lambda e, b=btn: b.configure(
                bg=self.colors["button_bg"]) if b["state"] != "disabled" else None)

    def cell_click(self, row, col):
        if not self.manual_control:
            return
        
        if self.selected_cell:
            prev_row, prev_col = self.selected_cell
            self.update_cell_color(prev_row, prev_col)

        self.selected_cell = (row, col)
        self.buttons[row][col].configure(bg=self.colors["selected"])
        self.game_frame.focus_set()
        
    def on_key_press(self, event):
        if not self.manual_control or not self.selected_cell:
            return
        
        try:
            if event.char.isdigit():
                num = int(event.char)
                if 0 <= num <= 9:
                    self.number_click(num)
                else:
                    messagebox.showwarning("Invalid Input", "Please enter a number between 0 and 9")
        except ValueError:
            pass

    def number_click(self, number):
        if not self.manual_control or not self.selected_cell:
            return

        row, col = self.selected_cell
        
        self.current_state[row][col] = number
        self.buttons[row][col].configure(text=str(number) if number != 0 else "")
        
        if (row, col) in self.ai_filled_numbers:
            self.ai_filled_numbers.remove((row, col))
            
        if number != 0:
            self.fixed_numbers.add((row, col))
        else:
            if (row, col) in self.fixed_numbers:
                self.fixed_numbers.remove((row, col))
        
        self.check_completion()
        
        if not self.check_conflicts(row, col):
            self.update_cell_color(row, col)
            self.selected_cell = None

    def start_model(self):
        if not self.is_model_running and self.manual_control:
            result, solution = solve_sudoku_gt(self.current_state)
            if result == 0:
                messagebox.showerror("Error", "No solution exists!")
                return
            elif result == 2:
                messagebox.showerror("Error", "Multiple solutions exist!")
                return
                
            self.is_model_running = True
            self.manual_control = False
            self.set_buttons_state("disabled")

            self.reasoning_text.configure(state="normal")
            self.reasoning_text.delete(1.0, tk.END)
            self.reasoning_text.configure(state="disabled")

            def recall(text, position=None, number=None, tokens=0):
                def update():
                    try:
                        if text:
                            self.update_reasoning(text, tokens)
                        if position is not None and number is not None:
                            row, col = position
                            if (row, col) in self.fixed_numbers:
                                print(f"Error: Cannot modify fixed number at ({row}, {col})\n")
                                return
                                
                            if self.current_state[row][col] != number:
                                self.current_state[row][col] = number
                                self.buttons[row][col].configure(
                                    text="" if number == 0 else str(number)
                                )
                                if number != 0:
                                    self.ai_filled_numbers.add((row, col))
                                elif (row, col) in self.ai_filled_numbers:
                                    self.ai_filled_numbers.remove((row, col))
                                
                                self.update_labels()
                                self.update_cell_color(row, col)
                                self.check_completion()
                    except Exception as e:
                        print(f"Error in recall: {str(e)}")

                self.master.after(0, update)

            def run_model():
                try:
                    self.model.solve(self, recall)
                except Exception as e:
                    print(f"Error in model: {str(e)}")
                finally:
                    def restore():
                        self.set_buttons_state("normal")
                        self.manual_control = True
                        self.is_model_running = False

                    self.master.after(0, restore)

            Thread(target=run_model, daemon=True).start()

    def new_game(self):
        if self.timer_id:
            self.master.after_cancel(self.timer_id)
            self.timer_id = None

        self.manual_control = True
        self.timer_running = False
        self.time = 0
        self.is_model_running = False
        self.selected_cell = None
        self.total_tokens = 0
        self.token_count.config(text="0")

        self.reasoning_text.configure(state="normal")
        self.reasoning_text.delete(1.0, tk.END)
        self.reasoning_text.configure(state="disabled")

        self.initial_state = generate_sudoku(difficulty=DEFAULT_DIFFICULTY, seed=SEED)
        self.current_state = copy.deepcopy(self.initial_state)
        
        self.fixed_numbers = set()
        self.ai_filled_numbers = set()
        for i in range(9):
            for j in range(9):
                if self.initial_state[i][j] != 0:
                    self.fixed_numbers.add((i, j))

        self.outer_frame.configure(bg=self.colors["glow_unsolved"])

        self.update_buttons()
        self.update_labels()

        self.manual_control = True
        self.timer_running = True
        self.update_timer()

    def update_timer(self):
        if self.timer_running:
            self.time += 1
            self.update_labels()
            self.timer_id = self.master.after(1000, self.update_timer)

    def update_labels(self):
        self.time_label.config(text=f"{self.time}s")

    def update_buttons(self):
        for i in range(9):
            for j in range(9):
                value = self.current_state[i][j]
                if value == 0:
                    self.buttons[i][j].configure(
                        text="",
                        bg=self.colors["empty"],
                        fg=self.colors["tile_text"])
                else:
                    self.update_cell_color(i, j)
                    self.buttons[i][j].configure(
                        text=str(value),
                        fg=self.colors["tile_text"]
                    )
                    
    def update_cell_color(self, row, col):
        value = self.current_state[row][col]
        if value == 0:
            self.buttons[row][col].configure(bg=self.colors["empty"])
            return

        has_conflict = self.check_conflicts(row, col)
        
        if has_conflict and self.show_conflicts:
            self.buttons[row][col].configure(bg=self.colors["error"])
        elif (row, col) in self.fixed_numbers:
            self.buttons[row][col].configure(bg=self.colors["fixed"])
        elif (row, col) in self.ai_filled_numbers:
            self.buttons[row][col].configure(bg=self.colors["tile"])
        else:
            self.buttons[row][col].configure(bg=self.colors["fixed"])

    def check_conflicts(self, row, col):
        value = self.current_state[row][col]
        if value == 0:
            return False

        for j in range(9):
            if j != col and self.current_state[row][j] == value:
                return True

        for i in range(9):
            if i != row and self.current_state[i][col] == value:
                return True

        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if (i != row or j != col) and self.current_state[i][j] == value:
                    return True

        return False

    def is_solved(self):
        for i in range(9):
            for j in range(9):
                if self.current_state[i][j] == 0:
                    return False
                if self.check_conflicts(i, j):
                    return False

        return True

    def check_completion(self):
        if self.is_solved():
            if self.timer_running:
                self.timer_running = False
                if self.timer_id:
                    self.master.after_cancel(self.timer_id)
                    self.timer_id = None
            
            self.outer_frame.configure(bg=self.colors["glow_solved"])
            return True
        
        self.outer_frame.configure(bg=self.colors["glow_unsolved"])
        if not self.timer_running and not self.is_model_running:
            self.timer_running = True
            self.update_timer()
        return False

    def on_hover(self, button):
        if button["state"] != "disabled" and not self.is_model_running:
            current_bg = button["bg"]
            if current_bg == self.colors["tile"]:
                button.configure(bg=self.colors["button_hover"])

    def on_leave(self, button):
        if button["state"] != "disabled" and not self.is_model_running:
            for i in range(9):
                for j in range(9):
                    if self.buttons[i][j] == button:
                        if (i, j) != self.selected_cell:
                            self.update_cell_color(i, j)
                        return

    def set_buttons_state(self, state):
        for i in range(9):
            for j in range(9):
                self.buttons[i][j].configure(state=state)

        for widget in self.game_container.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, tk.Button):
                        child.configure(state=state)

    def update_reasoning(self, text, tokens=1):
        self.reasoning_text.configure(state="normal")
        
        if text.startswith("> Fill cell"):
            self.reasoning_text.insert(tk.END, text, "move")
        else:
            self.reasoning_text.insert(tk.END, text)
            
        self.reasoning_text.see(tk.END)
        self.reasoning_text.configure(state="disabled")

        self.total_tokens += tokens
        self.token_count.config(text=str(self.total_tokens))
        

class RWKVModel:
    def __init__(self):

        os.environ["RWKV_JIT_ON"] = "1"
        os.environ["RWKV_CUDA_ON"] = "0"

        from rwkv_model import RWKV
        from rwkv.utils import PIPELINE, PIPELINE_ARGS
        from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

        self.model = RWKV(model=MODEL_PATH, strategy="cuda fp16", verbose=False)
        self.pipeline = PIPELINE(self.model, "rwkv_vocab_v20230424")
        self.pipeline.tokenizer = TRIE_TOKENIZER("sudoku_vocab.txt")
        self.gen_args = PIPELINE_ARGS(top_k=1, alpha_frequency=0, alpha_presence=0, token_stop=[105])

        self.model.forward([0, 1], None)
        self.model.forward([0], None)
        
        self.recall = None
        self.new_line = ''
        
    def my_callback(self, text):
        # print(text, end="", flush=True)
        self.new_line += text
        
        if text.endswith("\n"):
            # check if need to update board
            if self.new_line.startswith("> Fill cell"):
                self.new_line = self.new_line.strip()
                row = int(self.new_line[13])
                col = int(self.new_line[16])
                num = int(self.new_line[-1])
                self.recall('', (row, col), num, tokens=0)
            
            self.new_line = ''
            
        self.recall(text, tokens=1)
        
    def solve(self, puzzle, recall):
        
        self.recall = recall
        current_state = copy.deepcopy(puzzle.current_state)
        input_str = f"<input>\n{format_board(current_state)}\n</input>\n\n"
        print(input_str)
        
        self.pipeline.generate(input_str, token_count=10000000, args=self.gen_args, callback=self.my_callback)  


def generate_sudoku(difficulty, seed):
#     return [
#     [0, 0, 8, 1, 6, 7, 0, 2, 0],
#     [5, 0, 0, 2, 3, 0, 0, 0, 0],
#     [7, 6, 0, 0, 5, 4, 8, 0, 1],
#     [8, 7, 0, 0, 4, 0, 0, 0, 0],
#     [0, 2, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 4, 0, 0, 3, 0, 9, 0],
#     [0, 0, 0, 0, 0, 0, 3, 7, 0],
#     [0, 4, 0, 0, 0, 0, 0, 8, 0],
#     [3, 1, 0, 8, 0, 6, 9, 0, 4]
# ]
    sudoku, _ = gen(difficulty=difficulty, seed=seed)
    return sudoku


def main():
    root = tk.Tk()
    root.configure(bg="#1a1b26")
    game = ModernSudokuGame(root)
    root.mainloop()


if __name__ == "__main__":
    main()