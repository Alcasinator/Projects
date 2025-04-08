if direction == 1:  # Clockwise
            cube['U'] = np.rot90(cube['U'], -1)
            # Update adjacent faces (F, R, B, L) clockwise
            front_row = cube['F'][0].copy()
            right_col = cube['R'][:, 0].copy()
            back_row = cube['B'][0].copy()
            left_col = cube['L'][:, 2].copy()
            cube['F'][0] = left_col[::-1]  # Reverse to match orientation
            cube['R'][:, 0] = front_row
            cube['B'][0] = right_col[::-1]  # Reverse to match orientation
            cube['L'][:, 2] = back_row