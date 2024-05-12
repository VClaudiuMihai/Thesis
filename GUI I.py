import tkinter as tk
from tkinter import scrolledtext
import subprocess

def run_program(program):
    result = subprocess.run(["python", program], capture_output=True, text=True)
    output = result.stdout if result.stdout else result.stderr
    update_text_box(output)

def create_text_box_window():
    text_box_window = tk.Toplevel(window)
    text_box_window.title("Program Output")

    # Create a text box to display the program output
    text_box = scrolledtext.ScrolledText(text_box_window, height=20, width=60, font=("Arial", 12))
    text_box.pack(padx=10, pady=10)

    return text_box

def update_text_box(output):
    text_box.delete('1.0', tk.END)
    text_box.insert(tk.END, output)

window = tk.Tk()
window.title("Program Runner")

# Set the window size and position
window.geometry("600x800")
window.resizable(False, False)

# Create a frame to hold the buttons
button_frame = tk.Frame(window)
button_frame.pack(pady=20)

# Create buttons with improved style
button_styles = {
    "background": "#4CAF50",
    "foreground": "white",
    "font": ("Arial", 14),
    "width": 40,
    "height": 2,
    "bd": 0,
    "relief": tk.RAISED,
    "activebackground": "#45A049",
}

buttons = [
    ("Random Forest", "Random Forest I.py"),
    ("Ada Boost", "Ada Boost I.py"),
    ("Gradient Boosting", "Gradient Boosting Regressor I.py"),
    ("K-Neighbors", "K-Neighbors Regression I.py"),
    ("SVR", "Support Vector Regression I.py"),
    ("Graph", "Graph I.py"),
    ("Individual Graph", "Individual Graph I.py"),
    ("Compare", "Compare I.py"),
]

text_box = create_text_box_window()

# Create and pack the buttons
for button_text, program_file in buttons:
    button = tk.Button(button_frame, text=button_text, **button_styles, command=lambda file=program_file: run_program(file))
    button.pack(pady=5)

window.mainloop()
