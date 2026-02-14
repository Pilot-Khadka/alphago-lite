from alphago.display_board import GoBoardDisplay


def main():
    display = GoBoardDisplay(board_size=19)
    display.draw_board()
    display.run()
    pass


if __name__ == "__main__":
    main()
