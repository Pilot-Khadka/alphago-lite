from ..go.gotypes import Point


def is_point_an_eye(board, point, color):
    """
    eye is an empty point completely surrounded by friendly stones
    """
    grid = board._grid

    # point must be empty to be an eye
    if point in grid:
        return False

    # all adjacent points must be same color
    for neighbor in board._neighbors[point]:
        neighbor_string = grid.get(neighbor)
        if neighbor_string is None or neighbor_string.color != color:
            return False

    # check diagonal corner
    friendly_corners = 0
    off_board_corners = 0

    size = board.size
    for row in (point.row - 1, point.row + 1):
        for col in (point.col - 1, point.col + 1):
            if 1 <= row <= size and 1 <= col <= size:
                corner_string = grid.get(Point(row, col))
                if corner_string is not None and corner_string.color == color:
                    friendly_corners += 1
            else:
                off_board_corners += 1

    # must have strong diagonal support(>=3 friendly/off board corners)
    # i.e must control 3/4 corners if point is in middle of baord
    # if point is on edge, all corners
    if off_board_corners > 0:
        return off_board_corners + friendly_corners == 4
    return friendly_corners >= 3
