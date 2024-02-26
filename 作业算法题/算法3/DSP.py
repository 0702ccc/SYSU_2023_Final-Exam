def is_valid(arrangement, m, n):
    # 检查哨兵布置是否合法
    for i in range(m):
        for j in range(n):
            if arrangement[i][j] == 0:
                # 当前陈列室没有哨兵
                if i > 0 and arrangement[i - 1][j] == 1:
                    continue  # 上方陈列室没有哨兵
                if i < m - 1 and arrangement[i + 1][j] == 1:
                    continue  # 下方陈列室没有哨兵
                if j > 0 and arrangement[i][j - 1] == 1:
                    continue  # 左方陈列室没有哨兵
                if j < n - 1 and arrangement[i][j + 1] == 1:
                    continue  # 右方陈列室没有哨兵
                return False
    return True


def count_sentries(arrangement):
    # 计算哨兵的数量
    return sum(row.count(1) for row in arrangement)


count = 1
best_count = float('inf')  # 初始化为正无穷大，以确保第一次比较成功
best_arrangement = None


def generate_arrangements(arrangement, row, col, m, n):
    global count, best_count, best_arrangement
    count += 1
    # 计算剩余房间的下界估计
    remaining_rooms = (m - row - 1) * n + (n - col - 1)
    lower_bound = count_sentries(arrangement) + remaining_rooms / 5
    # 如果下界大于等于当前最佳值，进行剪枝
    if lower_bound >= best_count:
        return

    # 递归生成所有可能的哨兵布置情况
    if is_valid(arrangement, m, n):
        current_count = count_sentries(arrangement)
        if current_count < best_count:
            best_count = current_count
            best_arrangement = [row[:] for row in arrangement]

    if row == m - 1 and col == n - 1:
        return
    else:
        next_row = row + 1 if col == n - 1 else row
        next_col = 0 if col == n - 1 else col + 1

        # 不放置哨兵
        generate_arrangements(arrangement, next_row, next_col, m, n)

        # 放置哨兵
        arrangement[row][col] = 1
        generate_arrangements(arrangement, next_row, next_col, m, n)
        arrangement[row][col] = 0


def branch_and_bound(m, n):
    global best_arrangement
    # 初始化一个空的m x n数组，表示博物馆的陈列室
    arrangement = [[0] * n for _ in range(m)]
    best_arrangement = [row[:] for row in arrangement]  # Initialize with a copy

    # 生成所有可能的哨兵布置情况
    generate_arrangements(arrangement, 0, 0, m, n)
    return best_arrangement

count = 1
# 示例：5x5的博物馆
museum_arrangement = branch_and_bound(5, 5)
print(count)
# 打印最佳的哨兵布置方案
for row in museum_arrangement:
    print(row)
