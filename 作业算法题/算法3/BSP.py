from queue import PriorityQueue

# 判断房间布局是否有效的函数
def is_valid(arrangement, m, n):
    num = 0
    for i in range(m):
        for j in range(n):
            if arrangement[i][j] == 0:
                # 检查上方、下方、左边、右边是否有哨兵，若有则不合法
                if i > 0 and arrangement[i - 1][j] == 1:
                    continue
                if i < m - 1 and arrangement[i + 1][j] == 1:
                    continue
                if j > 0 and arrangement[i][j - 1] == 1:
                    continue
                if j < n - 1 and arrangement[i][j + 1] == 1:
                    continue
                num += 1
    return num

# 统计一个布局中哨兵的数量的函数
def count_sentries(arrangement):
    return sum(row.count(1) for row in arrangement)

# 估算哨兵数量下界的函数
def estimate_lower_bound(arrangement, m, n):
    remaining_rooms = is_valid(arrangement, m, n)
    return count_sentries(arrangement) + remaining_rooms / 5

# 使用BFS和分支定界法的主函数
def bfs_branch_and_bound(m, n):
    global count  # 用于计数已探索的节点数的变量
    count = 1
    best_count = float('inf')
    best_arrangement = None
    lowest = round(m * n / 3) + 2

    arrangement = [[0] * n for _ in range(m)]
    best_arrangement = [row[:] for row in arrangement]
    pq = PriorityQueue()
    pq.put((estimate_lower_bound(arrangement, m, n), arrangement, 0, 0))

    while not pq.empty():
        _, arrangement, row, col = pq.get()
        current_count = count_sentries(arrangement)
        lower_bound = estimate_lower_bound(arrangement, m, n)

        if lower_bound >= min(best_count, lowest):
            # 如果当前节点的下界不比全局最优解和最低估计值小，放弃此节点
            continue

        if is_valid(arrangement, m, n) == 0 and current_count < best_count:
            # 如果当前节点是合法解且哨兵数量更少，更新最优解
            best_count = current_count
            best_arrangement = [row[:] for row in arrangement]

        if row == m - 1 and col == n - 1:
            # 如果已到达最后一个房间，继续下一个节点
            continue
        else:
            next_row = row + 1 if col == n - 1 else row
            next_col = 0 if col == n - 1 else col + 1

            # 将下一个节点加入队列
            pq.put((estimate_lower_bound(arrangement, m, n), arrangement, next_row, next_col))
            arrangement[row][col] = 1  # 尝试在当前房间放置哨兵
            pq.put((estimate_lower_bound(arrangement, m, n), [row[:] for row in arrangement],
                    next_row, next_col))
            arrangement[row][col] = 0  # 回溯，尝试不在当前房间放置哨兵

        count += 1

    return best_arrangement

# 主程序
if __name__ == "__main__":
    with open("input.txt", "r") as input_file:
        m, n = map(int, input_file.readline().split())

    museum_arrangement = bfs_branch_and_bound(m, n)
    print("探索节点个数", count)
    with open("output.txt", "w") as output_file:
        output_file.write("最少哨兵数量： " + str(count_sentries(museum_arrangement)) + "\n")
        for row in museum_arrangement:
            output_file.write(" ".join(map(str, row)) + "\n")
