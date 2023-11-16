import math

class RelativeLocation:
    def __init__(self):
        self.h_last_position = (0, 0)
        self.h_current_position = (0, 0)

        self.t_last_position = (0, 0)
        self.t_current_position = (0, 0)

        self.tail_locations = set((0,0))
        self.tail_locations_list = [(0,0)]
        self.tail_change = None

    def update(self, s=None, head_position=None):
        self.h_last_position = self.h_current_position
        x, y = self.h_last_position
        if s == 'L':
            self.h_current_position = (x - 1, y)
        elif s == 'R':
            self.h_current_position = (x + 1, y)
        elif s == 'U':
            self.h_current_position = (x, y+1)
        elif s == 'D':
            self.h_current_position = (x, y-1)
        else:
            self.h_current_position = head_position

        x_dist = self.h_current_position[0] - self.t_current_position[0]
        y_dist = self.h_current_position[1] - self.t_current_position[1]
        l = (x_dist**2 + y_dist**2)**0.5
        
        # move tails if 2 more more distances away
        if l >= 2:
            self.t_last_position = self.t_current_position
            if l == 2:
                if abs(x_dist) == 2:
                    self.t_current_position = (self.t_last_position[0] + x_dist//2, self.t_last_position[1])
                else:
                    self.t_current_position = (self.t_last_position[0], self.t_last_position[1] + y_dist//2)
            elif l > 2 and l < 2.5:
                if abs(x_dist) == 2:
                    self.t_current_position = (self.t_last_position[0] + x_dist//2, self.t_last_position[1] + y_dist)
                else:
                    self.t_current_position = (self.t_last_position[0] + x_dist, self.t_last_position[1] + y_dist//2)
            else:
                self.t_current_position = (self.t_last_position[0] + x_dist // 2, self.t_last_position[1] + y_dist // 2)
            self.tail_locations.add(self.t_current_position)
            self.tail_locations_list.append(self.t_current_position)
        
if __name__ == '__main__':
    parts = 10
    snake = [RelativeLocation() for i in range(parts - 1)]
    with open('Day 9/bigdata.txt') as f:
        for line in f:
            d, n = line.strip().split(' ')
            for x in range(int(n)):
                for i, s in enumerate(snake):
                    if i == 0:
                        s.update(d)
                    if i > 0:
                        head_position = snake[i-1].t_current_position
                        s.update(head_position=head_position)
    # for s in snake:
    #     print(s.h_current_position)
    print('Solution: ', len(snake[-1].tail_locations))

