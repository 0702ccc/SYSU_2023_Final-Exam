def findlrs(s):
    n = len(s)
    dp = []
    row = [0] * (n + 1)
    for _ in range(n + 1):  # 创建二维动态规划表 dp
        dp.append(row)
    result = ""
    max_length = 0
    for i in range(1, n + 1):
        t= []
        for j in range(i + 1, n + 1):
            if s[i - 1] == s[j - 1]:
                # 如果当前字符匹配，检查是否能构成更长的重复但不重叠子字符串
                if dp[i - 1][j - 1] + 1 > j - i:
                    dp[i][j] = j - i
                else:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                # 如果 dp[i][j] 大于 max_length，更新最大长度和结果
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    result = s[i - max_length:i]
            t.append(dp[i][j])
        print(t)
    # print(dp)
    return result,dp


# 打印动态规划表
def print_dp(dp):
    for row in dp:
        print(row)


# 测试示例
str1 = "geeksforgeeks"
str2 = "aabaabaaba"
str3 = "banana"
str4 = "abcabcabc"
# result1, dp1 = findlrs(str1)
result2, dp2 = findlrs(str2)
# result3, dp3 = findlrs(str3)
# result4, dp4 = findlrs(str4)

# print("String 1:", result1)
# print_dp(dp1)

# print("String 2:", result2)
# print_dp(dp2)
#
# print("String 3:", result3)
# print_dp(dp3)
#
# print("String 4:", result4)
# print_dp(dp4)
