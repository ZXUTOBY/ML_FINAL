import sys
import tty
import termios

def read_single_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)  # read 1 character
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def filter_lines_to_file(input_file='input.txt', output_file='a.txt', start=0, end=10000):
    print("Instructions:")
    print("▶ Press [1] to KEEP the line.")
    print("▶ Press [any other key] to SKIP the line.")
    print("▶ Press [q] to quit early.\n")

    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'a', encoding='utf-8') as fout:
        for i, line in enumerate(fin):
            if i < start:
                continue
            if i >= end:
                break

            print(f"\nLine {i}: {line.strip()}")
            print("Press [1] to keep")

            key = read_single_key()
            if key == '1':
                fout.write(line)
            elif key.lower() == 'q':
                print("Quitting early.")
                break

    print(f"\nFinished processing lines {start} to {end - 1}.")



# Example usage:
filter_lines_to_file('1_all.txt', '1_filtered.txt', start=1400, end=1500)

