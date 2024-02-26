def check(nums, n, m, target):
    stu = 1  # 初始化学生数量为1
    current = 0
    for i in range(n):
        current += nums[i]
        if current > target:
            stu += 1
            current = nums[i]
    if stu > m:
        return False
    else:
        return True  # 如果分配的学生数量不超过m，返回True

def find_min(nums, n, m):
    if n < m:   # 若书本数量小于学生数量 无法分配
        return -1
    left, right = nums[n-1], sum(nums)  # 二分查找的上下界
    while left < right:
        mid = left + (right - left) // 2
        if check(nums, n, m, mid):
            right = mid
        else:
            left = mid + 1
    return left


# 测试案例
nums = [5, 10, 15, 20, 25, 30, 35]
n = 7
m = 4
ans = find_min(nums, n, m)
print(ans)
